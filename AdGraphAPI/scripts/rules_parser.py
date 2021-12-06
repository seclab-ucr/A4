import os
import json
from os import listdir
from os.path import isfile, join
from urlparse import urlparse
from adblockparser import AdblockRules
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target-dir', type=str)
parser.add_argument('--domain', type=str)
parser.add_argument('--strategy', type=str)
parser.add_argument('--parse-modified', action='store_true')
args = parser.parse_args()


def write_json(file_to_write, content_to_write):
    with open(file_to_write, 'w') as outfile:
        json.dump(content_to_write, outfile, indent=4)


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def read_rules(rule_file):
    with open(rule_file) as f:
        content = f.readlines()

    raw_rules = []

    for line in content:
        raw_rules.append(line.strip())

    return raw_rules


def match_url(base_domain, url_row, event_type):

    parsed_domain = urlparse(base_domain)
    domain = '{uri.netloc}'.format(uri=parsed_domain)

    if domain.startswith('www.'):
        domain = domain[4:]

    url = url_row.strip()
    if url.endswith(('\\', '/')):
        url = url[:-1]

    parsed_uri = urlparse(url)

    url_netloc = '{uri.netloc}'.format(uri=parsed_uri)

    if url_netloc.startswith('www.'):
        url_netloc = url_netloc[4:]

    if url_netloc == domain:
        third_party_check = False
    else:
        third_party_check = True

    if event_type == IFRAME_CONTENT:
        subdocument_check = True
    else:
        subdocument_check = False

    if event_type == SCRIPT_CONTENT:
        if third_party_check:
            adblock_rules = adblock_rules_script_third
            options = {'third-party': True, 'script': True,
                       'domain': domain, 'subdocument': subdocument_check}
        else:
            adblock_rules = adblock_rules_script
            options = {'script': True, 'domain': domain,
                       'subdocument': subdocument_check}

    elif event_type == IMAGE_CONTENT:
        if third_party_check:
            adblock_rules = adblock_rules_image_third
            options = {'third-party': True, 'image': True,
                       'domain': domain, 'subdocument': subdocument_check}
        else:
            adblock_rules = adblock_rules_image
            options = {'image': True, 'domain': domain,
                       'subdocument': subdocument_check}

    elif event_type == CSS_CONTENT:
        if third_party_check:
            adblock_rules = adblock_rules_css_third
            options = {'third-party': True, 'stylesheet': True,
                       'domain': domain, 'subdocument': subdocument_check}
        else:
            adblock_rules = adblock_rules_css
            options = {'stylesheet': True, 'domain': domain,
                       'subdocument': subdocument_check}

    elif event_type == XMLHTTP_CONTENT:
        if third_party_check:
            adblock_rules = adblock_rules_xmlhttp_third
            options = {'third-party': True, 'xmlhttprequest': True,
                       'domain': domain, 'subdocument': subdocument_check}
        else:
            adblock_rules = adblock_rules_xmlhttp
            options = {'xmlhttprequest': True, 'domain': domain,
                       'subdocument': subdocument_check}

    elif third_party_check:
        adblock_rules = adblock_rules_third
        options = {'third-party': True, 'domain': domain,
                   'subdocument': subdocument_check}

    else:
        adblock_rules = adblock_rules_domain
        options = {'domain': domain, 'subdocument': subdocument_check}

    return adblock_rules.should_block(url.strip(), options)


def get_ad_check(base_domain_url, url_row, event_type):
    ad_check = False
    if url_row.startswith('http'):
        if match_url(base_domain_url, url_row, event_type):
            ad_check = True
    elif url_row.startswith('about:blank') or url_row.startswith('javascript:'):
        ad_check = False
    else:
        # print 'Formatted URL expected: ' + url_row
        pass

    return ad_check


SCRIPT_CONTENT = 'NetworkScriptRequest'
IFRAME_CONTENT = 'NetworkIframeRequest'
IMAGE_CONTENT = 'NetworkImageRequest'
VIDEO_CONTENT = 'NetworkVideoRequest'
CSS_CONTENT = 'NetworkLinkRequest'
XMLHTTP_CONTENT = 'NetworkXMLHTTPRequest'


home_dir = os.getenv("HOME")
base_directory = home_dir + '/rendering_stream/'
# make sure to use the specific version of Easylist.
filter_lists_addr = base_directory + 'filterlists/'

easylist_file = filter_lists_addr + 'easylist.txt'
easyprivacy_file = filter_lists_addr + 'easyprivacy.txt'
warning_file = filter_lists_addr + 'warning.txt'
killer_file = filter_lists_addr + 'killer.txt'
fanboyannoyance_file = filter_lists_addr + 'fanboyannoyance.txt'
blockzilla_file = filter_lists_addr + 'blockzilla.txt'
peter_file = filter_lists_addr + 'peter.txt'
squid_file = filter_lists_addr + 'squid.txt'

raw_rules = read_rules(easylist_file) + read_rules(easyprivacy_file)


adblock_rules_script = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024,
                                    supported_options=['script', 'domain', 'subdocument'], skip_unsupported_rules=False)
adblock_rules_script_third = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024, supported_options=[
                                          'third-party', 'script', 'domain', 'subdocument'], skip_unsupported_rules=False)

adblock_rules_image = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024,
                                   supported_options=['image', 'domain', 'subdocument'], skip_unsupported_rules=False)
adblock_rules_image_third = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024, supported_options=[
                                         'third-party', 'image', 'domain', 'subdocument'], skip_unsupported_rules=False)

adblock_rules_css = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024, supported_options=[
                                 'stylesheet', 'domain', 'subdocument'], skip_unsupported_rules=False)
adblock_rules_css_third = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024, supported_options=[
                                       'third-party', 'stylesheet', 'domain', 'subdocument'], skip_unsupported_rules=False)

adblock_rules_xmlhttp = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024, supported_options=[
                                     'xmlhttprequest', 'domain', 'subdocument'], skip_unsupported_rules=False)
adblock_rules_xmlhttp_third = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024, supported_options=[
                                           'third-party', 'xmlhttprequest', 'domain', 'subdocument'], skip_unsupported_rules=False)

adblock_rules_third = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024*1024, supported_options=[
                                   'third-party', 'domain', 'subdocument'], skip_unsupported_rules=False)
adblock_rules_domain = AdblockRules(raw_rules, use_re2=True, max_mem=1024*1024 *
                                    1024, supported_options=['domain', 'subdocument'], skip_unsupported_rules=False)


timeline_files = os.listdir(args.target_dir)
timeline_files_set = set(timeline_files)

if args.parse_modified:
    timeline_file = 'modified_%s_' % args.strategy + args.domain + '.json'
else:
    timeline_file = args.domain + '.json'
print("Now processing: %s" % timeline_file)
timeline_fpath = '/'.join([args.target_dir + '/' + timeline_file])
updated_stream_file_to_write = '/'.join([args.target_dir, 'parsed_' + timeline_file])
rendering_stream = read_json(timeline_fpath)
url = rendering_stream['url']

# Create a json object here that contains AD/NONAD information about each event.
updated_rendering_stream = {}
updated_rendering_stream['url'] = url
updated_rendering_stream['timeline'] = []

for json_item in rendering_stream['timeline']:
    updated_json_item = json_item

    if json_item['event_type'] == SCRIPT_CONTENT or json_item['event_type'] == IMAGE_CONTENT or json_item['event_type'] == VIDEO_CONTENT or json_item['event_type'] == CSS_CONTENT or json_item['event_type'] == IFRAME_CONTENT or json_item['event_type'] == XMLHTTP_CONTENT:
        if json_item['request_url'].strip() == '':
            ad_check = False
        else:
            ad_check = get_ad_check(
                url, json_item['request_url'], json_item['event_type'])
    else:
        ad_check = False

    if ad_check:
        updated_json_item['ad_check'] = 'AD'
    else:
        updated_json_item['ad_check'] = 'NONAD'

    updated_rendering_stream['timeline'].append(updated_json_item)

write_json(updated_stream_file_to_write, updated_rendering_stream)
