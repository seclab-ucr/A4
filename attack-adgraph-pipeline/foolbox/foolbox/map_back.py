import os
import json
import numpy as np
import random

# THIS MODULE MAPS PERTURBATIONS FROM FEATURE-SPACE TO TIMELINE-SPACE (WE CONSIDER IT
# AS A EQUVALENCE OF WEBPAGE-SPACE)

HOME_DIR = os.getenv("HOME")
BASE_TIMELINE_DIR = HOME_DIR + '/rendering_stream/'
BASE_DEF_DIR = HOME_DIR + "/attack-adgraph-pipeline/def/"

TEST_MODE = 2

CATEGORICAL_FEATURE_IDX = {
    "FEATURE_NODE_CATEGORY",
    "FEATURE_FIRST_PARENT_TAG_NAME",
    "FEATURE_FIRST_PARENT_SIBLING_TAG_NAME",
    "FEATURE_SECOND_PARENT_TAG_NAME",
    "FEATURE_SECOND_PARENT_SIBLING_TAG_NAME"
}

UNUSED_FEATURE_IDX = {
    "DOMAIN_NAME",
    "NODE_ID",
    "FEATURE_KATZ_CENTRALITY",
    "FEATURE_FIRST_PARENT_KATZ_CENTRALITY",
    "FEATURE_SECOND_PARENT_KATZ_CENTRALITY"
}


def write_json(file_to_write, content_to_write):
    with open(file_to_write, 'w') as outfile:
        json.dump(content_to_write, outfile, indent=2)


def write_html(file_to_write, content_to_write):
    if isinstance(content_to_write, str):
        with open(file_to_write, 'w') as outfile:
            outfile.write(content_to_write)
    else:
        with open(file_to_write, 'wb') as outfile:
            html_dump = content_to_write.prettify().encode("utf-8")
            outfile.write(html_dump)


def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_largest_fname(path, filter_keyword=[".modified", "parsed_"]):
    largest_size = -1
    largest_fname, largest_fpath = "", ""
    for fname in os.listdir(path):
        should_fliter = False
        for kw in filter_keyword:
            if kw in fname:
                should_fliter = True
                break
        if should_fliter:
            continue
        fpath = os.path.join(path, fname)
        if os.path.getsize(fpath) > largest_size:
            largest_size = os.path.getsize(fpath)
            largest_fname = fname
            largest_fpath = fpath
    return largest_fname, largest_fpath


def generate_node_creation_event(node_id,
                                 actor_id,
                                 node_type,
                                 tag_name):
    creation_event = {}
    creation_event["event_type"] = "NodeCreation"
    creation_event["node_type"] = int(node_type)
    creation_event["tag_name"] = str(tag_name)
    creation_event["node_id"] = str(node_id)
    creation_event["actor_id"] = str(actor_id)
    return creation_event


def read_features(fpath):
    features = {}
    with open(fpath, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        url_id = r.split(",")[1]
        features[url_id] = r
    return features


def read_feature_def(unencoded_idx_fpath, encoded_idx_fpath):
    unencoded_feature_def = {}
    with open(unencoded_idx_fpath, 'r') as fin:
        unencoded_data = fin.readlines()
    for r in unencoded_data:
        r = r.strip()
        idx, feature_name = r.split(",")
        unencoded_feature_def[int(idx)] = feature_name

    encoded_feature_def = {}
    idx_to_feature_name_map = {}
    for f in list(CATEGORICAL_FEATURE_IDX):
        encoded_feature_def[f] = {}
    with open(encoded_idx_fpath, 'r') as fin:
        encoded_data = fin.readlines()
    for r in encoded_data:
        r = r.strip()
        idx, feature_name = r.split(",")
        if '=' in feature_name:
            # "1" prevents spliting on tag name itself
            name, val = feature_name.split("=", 1)
            if name in CATEGORICAL_FEATURE_IDX:
                encoded_feature_def[name][val] = len(encoded_feature_def[name])
            idx_to_feature_name_map[int(idx)] = name
        else:
            idx_to_feature_name_map[int(idx)] = feature_name

    return unencoded_feature_def, encoded_feature_def, idx_to_feature_name_map


def one_hot_encode_x(x, unencoded_feature_def, encoded_feature_def):
    encoded_x = []
    for i in range(len(x)):
        feature_name = unencoded_feature_def[i]
        if feature_name in UNUSED_FEATURE_IDX:
            continue
        if i == len(x) - 1:
            continue
        if feature_name in CATEGORICAL_FEATURE_IDX:
            for j in range(len(encoded_feature_def[feature_name])):
                if j == encoded_feature_def[feature_name][x[i]]:
                    encoded_x.append(float(1.0))
                else:
                    encoded_x.append(float(0.0))
        else:
            encoded_x.append(x[i])
    return encoded_x


def read_feature_stats(fname):
    feature_stats = {}
    with open(fname, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        feature_name, maxn, minn = r.split(",")
        feature_stats[feature_name] = [maxn, minn]
    return feature_stats


def normalize_x(x, encoded_feature_def, feature_stats, idx_to_feature_name_map):
    def scale_feature(val, stats):
        [maxn, minn] = stats
        val = float(val)
        maxn, minn = float(maxn), float(minn)
        if maxn == minn:
            return val
        return (val - minn) / (maxn - minn)

    normalized_x = []

    for i in range(len(x)):
        val = x[i]
        feature_name = idx_to_feature_name_map[i]
        if feature_name in feature_stats:
            stats = feature_stats[feature_name]
            scaled_val = scale_feature(val, stats)
            normalized_x.append(scaled_val)
        else:
            normalized_x.append(val)

    return normalized_x


def get_x_from_features(domain, target_url_id):
    features_fpath = BASE_TIMELINE_DIR + "features/" + domain + '.csv'
    features = read_features(features_fpath)
    x_str = features[target_url_id]
    x_lst = x_str.split(",")
    return x_lst


def run_cpp_feature_extractor(domain_url, working_dir, parse_modified, browser_id, final_domain, strategy):
    def execute_shell_command(cmd):
        os.system(cmd)

    cmd_lst = []
    cmd_lst.append("cd %s" % working_dir)
    if parse_modified:
        cmd_lst.append("sh test.sh %s parse-mod %d %s %s" % (domain_url, browser_id, final_domain, strategy))
    else:
        cmd_lst.append("sh test.sh %s parse-unmod %d %s %s" % (domain_url, browser_id, final_domain, strategy))
    cmd = ' && '.join(cmd_lst)
    print("Issuing shell command: " + cmd)
    execute_shell_command(cmd)


def compute_x_after_mapping_back(
    domain_url, 
    url_id, 
    modified_html, 
    original_html_fname,
    strategy,
    working_dir,
    browser_id,
    final_domain,
):
    write_html(BASE_TIMELINE_DIR + 'html/' + 'modified_%s_' % strategy + original_html_fname, modified_html)
    run_cpp_feature_extractor(domain_url, working_dir, parse_modified=True, browser_id=browser_id, final_domain=final_domain, strategy=strategy)
    new_x = get_x_from_features(domain_url, url_id)

    unencoded_feature_def, encoded_feature_def, idx_to_feature_name_map = read_feature_def(
        BASE_DEF_DIR + "unnormalized_feature_idx.csv",
        BASE_DEF_DIR + "trimmed_wo_class_feature_idx.csv"
    )
    one_hot_encoded_x = one_hot_encode_x(
        new_x, unencoded_feature_def, encoded_feature_def)

    feature_stats = read_feature_stats(
        BASE_DEF_DIR + "col_stats_for_unnormalization.csv")
    normalized_x = normalize_x(
        one_hot_encoded_x, encoded_feature_def, feature_stats, idx_to_feature_name_map)
    
    return np.array(normalized_x).astype(np.float), new_x
