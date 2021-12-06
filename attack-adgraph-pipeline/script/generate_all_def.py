import argparse

BASE_DEF_DIR = "../def/"
BASE_DATA_DIR = "../data/"

CATEGORICAL_FEATURE_NAME_TO_IDX = {
    "FEATURE_NODE_CATEGORY": 13,
    "FEATURE_FIRST_PARENT_TAG_NAME": 23,
    "FEATURE_FIRST_PARENT_SIBLING_TAG_NAME": 26,
    "FEATURE_SECOND_PARENT_TAG_NAME": 42,
    "FEATURE_SECOND_PARENT_SIBLING_TAG_NAME": 45,
}
CATEGORICAL_IDX_TO_FEATURE_NAME = {}
for name, idx in CATEGORICAL_FEATURE_NAME_TO_IDX.items():
    CATEGORICAL_IDX_TO_FEATURE_NAME[idx] = name


def read_one_hot_feature_list(
    ds_fpath,
):
    one_hot_features = {}
    for feature, _ in CATEGORICAL_FEATURE_NAME_TO_IDX.items():
        one_hot_features[feature] = set()
    with open(ds_fpath, 'r') as fin:
        ds = fin.readlines()
    del ds[0]
    for row in ds:
        features = row.strip().split(',')
        one_hot_features["FEATURE_NODE_CATEGORY"].add(
            features[CATEGORICAL_FEATURE_NAME_TO_IDX["FEATURE_NODE_CATEGORY"]])
        one_hot_features["FEATURE_FIRST_PARENT_TAG_NAME"].add(
            features[CATEGORICAL_FEATURE_NAME_TO_IDX["FEATURE_FIRST_PARENT_TAG_NAME"]])
        one_hot_features["FEATURE_FIRST_PARENT_SIBLING_TAG_NAME"].add(
            features[CATEGORICAL_FEATURE_NAME_TO_IDX["FEATURE_FIRST_PARENT_SIBLING_TAG_NAME"]])
        one_hot_features["FEATURE_SECOND_PARENT_TAG_NAME"].add(
            features[CATEGORICAL_FEATURE_NAME_TO_IDX["FEATURE_SECOND_PARENT_TAG_NAME"]])
        one_hot_features["FEATURE_SECOND_PARENT_SIBLING_TAG_NAME"].add(
            features[CATEGORICAL_FEATURE_NAME_TO_IDX["FEATURE_SECOND_PARENT_SIBLING_TAG_NAME"]])
    return one_hot_features


def read_base_feature_def(
    base_def_fpath
):
    with open(base_def_fpath, 'r') as fin:
        data = fin.readlines()
    feature_info = {}
    for row in data:
        row = row.strip()
        if len(row.split(',')) == 3:
            feature_idx, feature_name, feature_type = row.split(',')
            info = {
                "id": feature_idx,
                "type": feature_type,
            }
        elif len(row.split(',')) == 4:
            feature_idx, feature_name, feature_type, dummy_value = row.split(
                ',')
            info = {
                "id": feature_idx,
                "type": feature_type,
                "dummy_value": dummy_value
            }
        feature_info[feature_name] = info
    return feature_info


def generate_preprocessing_feature_type_def(
    base_def,
    one_hot_features,
    preprocessing_feature_type_def_fpath
):
    preprocessing_feature_type_def_dict = {}
    preprocessing_feature_type_def = []
    for name, info in base_def.items():
        if name in CATEGORICAL_FEATURE_NAME_TO_IDX:
            preprocessing_feature_type_def_dict[info["id"]] = {
                "name": name,
                "type": ["C"] * len(one_hot_features[name]),
            }
        else:
            preprocessing_feature_type_def_dict[info["id"]] = {
                "name": name,
                "type": info["type"],
            }
    for _, info in sorted(preprocessing_feature_type_def_dict.items(), key=lambda item: int(item[0])):
        preprocessing_feature_type_def.append(info["type"])
    with open(preprocessing_feature_type_def_fpath, 'w') as fout:
        for type_def in preprocessing_feature_type_def:
            if isinstance(type_def, list):
                for t in type_def:
                    fout.write(t.lower() + '\n')
                continue
            if type_def == 'FF':
                type_def = 'F'
            if type_def == 'D' or type_def == 'L':
                continue
            fout.write(type_def.lower() + '\n')
    return preprocessing_feature_type_def


def generate_unnormalized_feature_idx(
    base_feature_info,
    unnormalized_feature_idx_fpath
):
    unnormalized_feature_idx = []
    for name, info in sorted(base_feature_info.items(), key=lambda item: int(item[1]['id'])):
        feature_id = info['id']
        unnormalized_feature_idx.append(feature_id + ',' + name + '\n')
    with open(unnormalized_feature_idx_fpath, 'w') as fout:
        fout.writelines(unnormalized_feature_idx)


def generate_trimmed_wo_class_feature_idx(
    base_feature_info,
    one_hot_info,
    trimmed_wo_class_feature_idx_fpath,
    trimmed_w_class_feature_idx_fpath,
    untrimmed_wo_class_feature_idx_fpath,
    untrimmed_w_class_feature_idx_fpath,
):
    trimmed_wo_class_feature_idx = []
    trimmed_w_class_feature_idx = []
    untrimmed_wo_class_feature_idx = []
    untrimmed_w_class_feature_idx = []

    for name, info in sorted(base_feature_info.items(), key=lambda item: int(item[1]['id'])):
        feature_type = info['type']
        if name not in CATEGORICAL_FEATURE_NAME_TO_IDX:
            if feature_type not in {'L', 'D'}:
                line = ','.join([
                    str(len(trimmed_wo_class_feature_idx)),
                    name
                ])
                trimmed_wo_class_feature_idx.append(line + '\n')
            if feature_type != 'D':
                line = ','.join([
                    str(len(trimmed_w_class_feature_idx)),
                    name
                ])
                trimmed_w_class_feature_idx.append(line + '\n')
            if feature_type != 'L':
                line = ','.join([
                    str(len(untrimmed_wo_class_feature_idx)),
                    name
                ])
                untrimmed_wo_class_feature_idx.append(line + '\n')
            line = ','.join([
                str(len(untrimmed_w_class_feature_idx)),
                name
            ])
            untrimmed_w_class_feature_idx.append(line + '\n')
        elif name in CATEGORICAL_FEATURE_NAME_TO_IDX:
            one_hot_token_list = sorted(list(one_hot_info[name]))
            for one_hot_token in one_hot_token_list:
                if feature_type not in {'L', 'D'}:
                    line = ','.join([
                        str(len(trimmed_wo_class_feature_idx)),
                        name + '=' + one_hot_token
                    ])
                    trimmed_wo_class_feature_idx.append(line + '\n')
                if feature_type != 'D':
                    line = ','.join([
                        str(len(trimmed_w_class_feature_idx)),
                        name + '=' + one_hot_token
                    ])
                    trimmed_w_class_feature_idx.append(line + '\n')
                if feature_type != 'L':
                    line = ','.join([
                        str(len(untrimmed_wo_class_feature_idx)),
                        name + '=' + one_hot_token
                    ])
                    untrimmed_wo_class_feature_idx.append(line + '\n')
                line = ','.join([
                    str(len(untrimmed_w_class_feature_idx)),
                    name + '=' + one_hot_token
                ])
                untrimmed_w_class_feature_idx.append(line + '\n')
    with open(trimmed_wo_class_feature_idx_fpath, 'w') as fout:
        fout.writelines(trimmed_wo_class_feature_idx)
    with open(trimmed_w_class_feature_idx_fpath, 'w') as fout:
        fout.writelines(trimmed_w_class_feature_idx)
    with open(untrimmed_wo_class_feature_idx_fpath, 'w') as fout:
        fout.writelines(untrimmed_wo_class_feature_idx)
    with open(untrimmed_w_class_feature_idx_fpath, 'w') as fout:
        fout.writelines(untrimmed_w_class_feature_idx)

    return trimmed_wo_class_feature_idx, untrimmed_wo_class_feature_idx


def generate_hand_preprocessing_defs(
    base_feature_info,
    col_stats_fpath,
    trimmed_wo_class_feature_idx,
    untrimmed_wo_class_feature_idx,
    hand_preprocessing_defs_fpath,
    hand_preprocessing_defs_for_remote_model_fpath
):
    def _gen_records_base_on_reference(
        reference_list,
        base_feature_info,
        col_stats_dict
    ):
        records = []

        for row in reference_list:
            row = row.strip()

            if row == '':
                continue
            _, feature_name = row.split(',')

            if '=' in feature_name:
                feature_name, one_hot_token = feature_name.split('=', 1)
            feature_type = base_feature_info[feature_name]['type']

            if feature_type in {'F', 'FF'}:
                maxn, minn = col_stats_dict[feature_name]
                record = ','.join([feature_type, maxn, minn]) + '\n'
            elif feature_type == 'B':
                record = 'B\n'
            elif feature_type == 'C':
                record = ','.join([feature_type, one_hot_token]) + '\n'
            elif feature_type == 'D':
                record = ','.join(
                    [feature_type, base_feature_info[feature_name]['dummy_value']]) + '\n'
            else:
                print("Wrong type: %s" % feature_type)
                raise Exception
            records.append(record)

        return records

    with open(col_stats_fpath, 'r') as fin:
        data = fin.readlines()
    col_stats_dict = {}
    for row in data:
        row = row.strip()
        if row == "":
            continue
        feature_name, maxn, minn = row.split(',')
        col_stats_dict[feature_name] = [maxn, minn]

    records1 = _gen_records_base_on_reference(
        trimmed_wo_class_feature_idx,
        base_feature_info,
        col_stats_dict
    )
    with open(hand_preprocessing_defs_for_remote_model_fpath, 'w') as fout:
        fout.writelines(records1)

    records2 = _gen_records_base_on_reference(
        untrimmed_wo_class_feature_idx,
        base_feature_info,
        col_stats_dict
    )
    with open(hand_preprocessing_defs_fpath, 'w') as fout:
        fout.writelines(records2)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse arguments (see description in source file)')
    parser.add_argument('--unnormalized-dataset', type=str)
    parser.add_argument('--base-feature-def', type=str)
    parser.add_argument('--preprocessing-feature-type-def-fpath', type=str,
                        default="preprocessed_adgraph_alexa_10k_feature_defs.txt")
    parser.add_argument('--unnormalized-feature-idx-fpath', type=str,
                        default="unnormalized_feature_idx.csv")
    parser.add_argument('--trimmed-wo-class-feature-idx-fpath', type=str,
                        default="trimmed_wo_class_feature_idx.csv")
    parser.add_argument('--trimmed-w-class-feature-idx-fpath', type=str,
                        default="trimmed_w_class_feature_idx.csv")
    parser.add_argument('--untrimmed-wo-class-feature-idx-fpath', type=str,
                        default="untrimmed_wo_class_feature_idx.csv")
    parser.add_argument('--untrimmed-w-class-feature-idx-fpath', type=str,
                        default="untrimmed_w_class_feature_idx.csv")
    parser.add_argument('--hand-preprocessing-defs-fpath', type=str,
                        default="hand_preprocessing_defs.csv")
    parser.add_argument('--hand-preprocessing-defs-for-remote-model-fpath', type=str,
                        default="hand_preprocessing_defs_for_remote_model.csv")
    parser.add_argument('--col-stats-fpath', type=str,
                        default="col_stats_for_unnormalization.csv")
    args = parser.parse_args()

    base_feature_info = read_base_feature_def(
        BASE_DEF_DIR + args.base_feature_def
    )
    one_hot_features = read_one_hot_feature_list(
        BASE_DATA_DIR + args.unnormalized_dataset,
    )
    preprocessing_feature_type_def = generate_preprocessing_feature_type_def(
        base_feature_info,
        one_hot_features,
        BASE_DEF_DIR + args.preprocessing_feature_type_def_fpath
    )
    generate_unnormalized_feature_idx(
        base_feature_info,
        BASE_DEF_DIR + args.unnormalized_feature_idx_fpath
    )
    trimmed_wo_class_feature_idx, untrimmed_wo_class_feature_idx = generate_trimmed_wo_class_feature_idx(
        base_feature_info,
        one_hot_features,
        BASE_DEF_DIR + args.trimmed_wo_class_feature_idx_fpath,
        BASE_DEF_DIR + args.trimmed_w_class_feature_idx_fpath,
        BASE_DEF_DIR + args.untrimmed_wo_class_feature_idx_fpath,
        BASE_DEF_DIR + args.untrimmed_w_class_feature_idx_fpath
    )
    generate_hand_preprocessing_defs(
        base_feature_info,
        BASE_DEF_DIR + args.col_stats_fpath,
        trimmed_wo_class_feature_idx,
        untrimmed_wo_class_feature_idx,
        BASE_DEF_DIR + args.hand_preprocessing_defs_fpath,
        BASE_DEF_DIR + args.hand_preprocessing_defs_for_remote_model_fpath
    )
