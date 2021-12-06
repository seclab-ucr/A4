import mimetypes
import os
import re
import typing
import urllib.parse
from urllib.parse import urlparse
from pathlib import Path

from werkzeug.security import safe_join

from mitmproxy import ctx, exceptions, flowfilter, http, version
from mitmproxy.utils.spec import parse_spec

HOME_DIR = os.getenv("HOME")


class MapLocalSpec(typing.NamedTuple):
    matches: flowfilter.TFilter
    regex: str
    local_path: Path


def parse_map_local_spec(option: str) -> MapLocalSpec:
    filter, regex, replacement = parse_spec(option)

    try:
        re.compile(regex)
    except re.error as e:
        raise ValueError(f"Invalid regular expression {regex!r} ({e})")

    try:
        path = Path(replacement).expanduser().resolve(strict=True)
    except FileNotFoundError as e:
        raise ValueError(f"Invalid file path: {replacement} ({e})")

    return MapLocalSpec(filter, regex, path)


def _safe_path_join(root: Path, untrusted: str) -> Path:
    """Join a Path element with an untrusted str.

    This is a convenience wrapper for werkzeug's safe_join,
    raising a ValueError if the path is malformed."""
    untrusted_parts = Path(untrusted).parts
    joined = safe_join(
        root.as_posix(),
        *untrusted_parts
    )
    if joined is None:
        raise ValueError("Untrusted paths.")
    return Path(joined)


def file_candidates(url: str, spec: MapLocalSpec) -> typing.List[Path]:
    """
    Get all potential file candidates given a URL and a mapping spec ordered by preference.
    This function already assumes that the spec regex matches the URL.
    """
    m = re.search(spec.regex, url)
    assert m
    if m.groups():
        suffix = m.group(1)
    else:
        suffix = re.split(spec.regex, url, maxsplit=1)[1]
        suffix = suffix.split("?")[0]  # remove query string
        suffix = suffix.strip("/")

    if suffix:
        decoded_suffix = urllib.parse.unquote(suffix)
        suffix_candidates = [decoded_suffix, f"{decoded_suffix}/index.html"]

        escaped_suffix = re.sub(r"[^0-9a-zA-Z\-_.=(),/]", "_", decoded_suffix)
        if decoded_suffix != escaped_suffix:
            suffix_candidates.extend([escaped_suffix, f"{escaped_suffix}/index.html"])
        try:
            return [
                _safe_path_join(spec.local_path, x)
                for x in suffix_candidates
            ]
        except ValueError:
            return []
    else:
        return [spec.local_path / "index.html"]


class MapLocal:
    def __init__(self):
        self.replacements: typing.List[MapLocalSpec] = []
        self.non_regex_html_replacements: typing.Dict[str, str] = {}
        self.non_regex_url_replacements: typing.Dict[str, str] = {}

    def load(self, loader):
        loader.add_option(
            "map_local", typing.Sequence[str], [],
            """
            Map remote resources to a local file using a pattern of the form
            "[/flow-filter]/url-regex/file-or-directory-path", where the
            separator can be any character.
            """
        )
        # wzj
        loader.add_option(
            "map_local_file", typing.Optional[str], None,
            """
            A file containing the mapping between url and local path.
            Only single page url to local file mapping is allowed.
            Example: 
                www.cnn.com, /home/user/www.cnn.com/index.html
            """
        )
        loader.add_option(
            "url_map_local_file", typing.Optional[str], None,
            """
            A file containing the mapping between original and perturbed url.
            Only single page url to local file mapping is allowed.
            Example:
                www.cnn.com, www.cnn.com/1.js
            """
        )
        loader.add_option(
            "use_modified", bool, False,
            """
            This flag specifies whether modified local files should
            be mapped to.
            """
        )
        loader.add_option(
            "eval_mode", bool, False,
            """
            This flag specifies whether we are now evaluating the
            web pages.
            """
        )
        loader.add_option(
            "url_id", typing.Optional[str], None, 
            """
            This specifies the URL ID.
            """
        )


    def configure(self, updated):
        if "map_local" in updated:
            print(ctx.options.map_local)
            for option in ctx.options.map_local:
                try:
                    spec = parse_map_local_spec(option)
                except ValueError as e:
                    raise exceptions.OptionsError(f"Cannot parse map_local option {option}: {e}") from e

                self.replacements.append(spec)
        # wzj
        if "map_local_file" in updated:
            if os.path.exists(ctx.options.map_local_file):
                f = open(ctx.options.map_local_file, 'r')
                for line in f:
                    line = line.strip()
                    original_domain, final_url = line.split(',', 1)
                    if ctx.options.use_modified:
                        if ctx.options.eval_mode:
                            original_domain, strategy = original_domain.split('_')
                            local_path = HOME_DIR + '/rendering_stream/eval_html/%s_URL_%s_%s.html' % (original_domain, ctx.options.url_id, strategy)
                        else:
                            local_path = HOME_DIR + '/rendering_stream/html/modified_' + original_domain + '.html'
                    else:
                        if ctx.options.eval_mode:
                            local_path = HOME_DIR + '/rendering_stream/eval_html/original_' + original_domain + '.html'
                        else:
                            local_path = HOME_DIR + '/rendering_stream/html/' + original_domain + '.html'
                    print("Adding HTML mapping: %s --> %s" % (final_url, local_path))
                    self.non_regex_html_replacements[final_url] = local_path
                f.close()
            else:
                raise exceptions.OptionsError("Cannot find map_local_file: %s" % ctx.options.map_local_file)
        # zst
        if "url_map_local_file" in updated:
            if os.path.exists(ctx.options.url_map_local_file):
                f = open(ctx.options.url_map_local_file, 'r')
                for line in f:
                    line = line.strip()
                    original_url, perturbed_url = line.split(',', 1)
                    original_scheme = urlparse(original_url)[0]
                    perturbed_components = urlparse(perturbed_url)
                    if perturbed_components[0] == '':
                        perturbed_components._replace(scheme=original_scheme)
                        perturbed_url = perturbed_components.geturl()
                    print("Adding URL mapping: %s --> %s" % (perturbed_url, original_url))
                    self.non_regex_url_replacements[perturbed_url] = original_url
                f.close()
            else:
                raise exceptions.OptionsError("Cannot find url_map_local_file: %s" % ctx.options.url_map_local_file)

    def request(self, flow: http.HTTPFlow) -> None:
        if flow.reply and flow.reply.has_message:
            return

        url = flow.request.pretty_url

        all_candidates = []
        for spec in self.replacements:
            if spec.matches(flow) and re.search(spec.regex, url):
                if spec.local_path.is_file():
                    candidates = [spec.local_path]
                else:
                    candidates = file_candidates(url, spec)
                all_candidates.extend(candidates)

                local_file = None
                for candidate in candidates:
                    if candidate.is_file():
                        local_file = candidate
                        break

                if local_file:
                    headers = {
                        "Server": version.MITMPROXY
                    }
                    mimetype = mimetypes.guess_type(str(local_file))[0]
                    if mimetype:
                        headers["Content-Type"] = mimetype

                    try:
                        contents = local_file.read_bytes()
                    except OSError as e:
                        ctx.log.warn(f"Could not read file: {e}")
                        continue

                    flow.response = http.HTTPResponse.make(
                        200,
                        contents,
                        headers
                    )
                    # only set flow.response once, for the first matching rule
                    return
        
        if url in self.non_regex_html_replacements:
            local_path = self.non_regex_html_replacements[url]
            ctx.log.info("Serving %s with local file %s" % (url, local_path))
            all_candidates.append(local_path)
            if os.path.isfile(local_path):
                headers = {
                    "Server": version.MITMPROXY
                }
                mimetype = mimetypes.guess_type(local_path)[0]
                if mimetype:
                    headers["Content-Type"] = mimetype

                try:
                    with open(local_path) as f:
                        contents = f.read()
                except OSError as e:
                    ctx.log.warn(f"Could not read file: {e}")
                    return

                flow.response = http.HTTPResponse.make(
                    200,
                    contents,
                    headers
                )
                # only set flow.response once, for the first matching rule
                return

        if url in self.non_regex_url_replacements:
            local_path = self.non_regex_url_replacements[url]
            ctx.log.info("Serving %s with original URL %s" % (url, local_path))

            flow.request.url = self.non_regex_url_replacements[url]

            return


        if all_candidates:
            flow.response = http.HTTPResponse.make(404)
            ctx.log.info(f"None of the local file candidates exist: {', '.join(str(x) for x in all_candidates)}")
