#!/usr/bin/env python
# coding: utf-8

# @Shitong Zhu, SRA 2019
# This script preprocess the raw dataset of AdGraph and transforms
# it into the format that is compatible with Keras training/testing
# and Foolbox attacks

# Required args:
# --rescale-feature-names: str, sepcifies the csv file containing feature names that need to be resacled
# --use-precomputed-stats: bool, determines whether we should use pre-computed stats
# to preprocess instead of computing them on the fly
# --original-ds: str, specifies the dataset file that has not been preprocessed
# --dump-untrimmed-label: bool
# --dump-untrimmed-no-label: bool
# --dump-trimmed-label: bool
# --dump-trimmed-no-label: bool

import pandas as pd
import numpy as np
from sklearn import preprocessing

import math

import argparse

UNUSED_COL_NAMES = [
    "DOMAIN_NAME",
    "NODE_ID",
    "FEATURE_KATZ_CENTRALITY",
    "FEATURE_FIRST_PARENT_KATZ_CENTRALITY",
    "FEATURE_SECOND_PARENT_KATZ_CENTRALITY"
]
CLASS_COL_NAME = ["CLASS"]
UNUSED_COL_NAMES_CLASS = [
    "DOMAIN_NAME",
    "NODE_ID",
    "FEATURE_KATZ_CENTRALITY",
    "FEATURE_FIRST_PARENT_KATZ_CENTRALITY",
    "FEATURE_SECOND_PARENT_KATZ_CENTRALITY",
    "CLASS"
]
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

LABEL = {"AD": "1", "NONAD": "0"}

parser = argparse.ArgumentParser(
    description='Parse arguments (see description in source file)')

parser.add_argument('--rescale-feature-names', type=str)
parser.add_argument('--precomputed-stats', type=str)
parser.add_argument('--original-ds', type=str)
parser.add_argument('--suffix', type=str, default="")

parser.add_argument('--unnormalized-feature-idx-fpath',
                    type=str, default='unnormalized_feature_idx.csv')
parser.add_argument('--untrimmed-w-class-feature-idx-fpath',
                    type=str, default='untrimmed_w_class_feature_idx.csv')
parser.add_argument('--one-hot-info-fpath',
                    type=str, default='one_hot_info.csv')

parser.add_argument('--dump-untrimmed-label',
                    dest='dump_untrimmed_label', action='store_true')
parser.add_argument('--dump-untrimmed-no-label',
                    dest='dump_untrimmed_no_label', action='store_true')
parser.add_argument('--dump-trimmed-label',
                    dest='dump_trimmed_label', action='store_true')
parser.add_argument('--dump-trimmed-no-label',
                    dest='dump_trimmed_no_label', action='store_true')

args = parser.parse_args()


def read_dataset(fname, convert_label=True, debug=False):
    def convert_from_str_to_int(str):
        return LABEL[str]

    dataframe = pd.read_csv(fname)
    if convert_label:
        dataframe["CLASS"] = dataframe["CLASS"].apply(convert_from_str_to_int)
    if debug:
        print(dataframe[:10])
    with open(fname, 'r') as fin:
        header = fin.readline()
    return dataframe, header


def get_col_stats(dataframe, cols):
    col_stats = {}
    for col in cols:
        col_stats[col] = []
        col_stats[col].append(dataframe[col].max())
        col_stats[col].append(dataframe[col].min())
    return col_stats


def rescale_dataframe(dataframe, cols):
    scaled_df = dataframe.copy()
    scaler = preprocessing.MinMaxScaler()
    scaled_df[cols] = scaler.fit_transform(scaled_df[cols])
    return scaled_df


def hand_rescale_dataframe(dataframe, col_stats, debug=False):
    def rescale_a_value(val, col_stats):
        maxn, minn = col_stats
        maxn, minn = float(maxn), float(minn)
        val = float(val)
        if maxn == minn:
            rescaled_val = val
        else:
            rescaled_val = (val - minn) / (maxn - minn)
        return rescaled_val

    scaled_df = dataframe.copy()
    for col_name, stats in col_stats.items():
        if debug:
            print(col_name, stats)
        scaled_df[col_name] = scaled_df[col_name].apply(
            rescale_a_value, col_stats=stats)
    return scaled_df


def one_hot_encode_x(x, unencoded_feature_def, encoded_feature_def, debug=False):
    def check_if_missing(val):
        if isinstance(x[i], float) and math.isnan(x[i]):
            return True
        if isinstance(x[i], str) and x[i] == "?":
            return True
        return False

    encoded_x = []
    cnt = 0
    if debug:
        print("Length of X:", len(x))
        print(x)
    for i in range(len(x)):
        feature_name = unencoded_feature_def[i]
        if feature_name in CATEGORICAL_FEATURE_IDX:
            # handle missing value
            if check_if_missing(x[i]):
                # replace the missing value with a default value
                if debug:
                    print(list(encoded_feature_def[feature_name].keys()))
                    input("Press Enter to continue...")
                x[i] = '?'
            for j in range(len(encoded_feature_def[feature_name])):
                if debug:
                    print(feature_name, x[i],
                          encoded_feature_def[feature_name])
                # print(j, encoded_feature_def[feature_name], x[i])
                # input("...")
                if j == encoded_feature_def[feature_name][x[i]]:
                    encoded_x.append(float(1.0))
                    # input("Hit")
                else:
                    encoded_x.append(float(0.0))
                cnt += 1

        else:
            encoded_x.append(x[i])
    if debug:
        print(cnt)
        print(len(encoded_x))
        input("Press Enter to continue...")
    return encoded_x


# this method is very slow for some unknown reason...
def one_hot_encode(dataframe, unencoded_feature_def, encoded_feature_def, encoded_col_names, debug=True):
    encoded_df = pd.DataFrame(columns=encoded_col_names)
    for i, row in dataframe.iterrows():
        if debug:
            print("[INFO] encoding #%d..." % i)
        encoded_row = one_hot_encode_x(
            row.values, unencoded_feature_def, encoded_feature_def, debug=True)
        row_series = pd.Series(encoded_row, index=encoded_col_names)
        encoded_df = encoded_df.append(row_series, ignore_index=True)
    return encoded_df


# let's try first generating the entire array and then convert it
# to a dataframe
def one_hot_encode_2(dataframe, unencoded_feature_def, encoded_feature_def, encoded_col_names, debug=False):
    if debug:
        print("Length of colnames:", len(encoded_col_names))
        print("Length of unencoded:", len(unencoded_feature_def))
        print("Length of encoded:", len(encoded_feature_def))
        cnt = 0
        for k, v in encoded_feature_def.items():
            for k1, v1 in v.items():
                cnt += 1
        print(cnt)
        input("Press Enter to continue...")
    df_np = dataframe.to_numpy()
    encoded_df_np = []
    for i in range(len(df_np)):
        if debug:
            print("[INFO] encoding #%d..." % i)
        encoded_row = one_hot_encode_x(
            df_np[i], unencoded_feature_def, encoded_feature_def
        )
        encoded_df_np.append(encoded_row)
    encoded_df = pd.DataFrame(encoded_df_np, columns=encoded_col_names)
    return encoded_df


def read_col_names_to_rescale(fname):
    with open(fname, 'r') as fin:
        data = fin.readlines()
    col_names = []
    for r in data:
        if r.startswith("##RESCALE_FEATURES"):
            continue
        elif r.startswith("##FLOAT_FEATURES"):
            break
        else:
            col_names.append(r)
    col_names = list(map(lambda l: l.strip(), col_names))
    return col_names


def read_col_stats(fname):
    col_stats = {}
    with open(fname, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        col_name, maxn, minn = r.split(',')
        col_stats[col_name] = [maxn, minn]
    return col_stats


def dump_col_stats(fname, col_stats):
    with open(fname, 'w') as fout:
        for col_name, stats in col_stats.items():
            # stats[0] is maximum and [1] is minimum
            stats_str = ','.join([col_name, str(stats[0]), str(stats[1])])
            fout.write(stats_str + '\n')


def dump_dataset(fname, dataframe):
    dataframe.to_csv(fname, index=None)


def remove_cols_from_dataframe(dataframe, col_names, debug=False):
    removed_df = dataframe.copy()
    if debug:
        for col in col_names:
            print(removed_df[col])
    removed_df.drop(columns=col_names, inplace=True)
    return removed_df


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
    encoded_col_names = []

    for f in list(CATEGORICAL_FEATURE_IDX):
        encoded_feature_def[f] = {}
    with open(encoded_idx_fpath, 'r') as fin:
        encoded_data = fin.readlines()
    for r in encoded_data:
        r = r.strip()
        idx, feature_name = r.split(",")
        encoded_col_names.append(feature_name)
        if '=' in feature_name:
            # "1" prevents spliting on tag name itself
            name, val = feature_name.split("=", 1)
            if name in CATEGORICAL_FEATURE_IDX:
                encoded_feature_def[name][val] = len(encoded_feature_def[name])
            idx_to_feature_name_map[int(idx)] = name
        else:
            idx_to_feature_name_map[int(idx)] = feature_name

    return unencoded_feature_def, encoded_feature_def, idx_to_feature_name_map, encoded_col_names


cols_to_rescale = read_col_names_to_rescale(
    "../def/%s" % args.rescale_feature_names)

# unnormalized_target_w_unused_numbered_class_gt.csv
df, header = read_dataset("../data/%s" % args.original_ds, convert_label=False)

df.replace(np.NaN, '', inplace=True)

unencoded_feature_def, encoded_feature_def, idx_to_feature_name_map, encoded_col_names = read_feature_def(
    "../def/%s" % args.unnormalized_feature_idx_fpath,
    "../def/%s" % args.untrimmed_w_class_feature_idx_fpath
)

df = one_hot_encode_2(
    df, unencoded_feature_def, encoded_feature_def, encoded_col_names)
print("[INFO] Done: one-hot-encoding")

if args.precomputed_stats:
    col_stats = read_col_stats("../def/%s" % args.precomputed_stats)
    untrimmed_scaled_df_w_labels = hand_rescale_dataframe(df, col_stats)
else:
    col_stats = get_col_stats(df, cols_to_rescale)
    untrimmed_scaled_df_w_labels = rescale_dataframe(df, cols_to_rescale)
    dump_col_stats("../def/col_stats_for_unnormalization.csv", col_stats)
print("[INFO] Done: normalization")

if args.dump_untrimmed_label:
    if not args.precomputed_stats:
        fname = "untrimmed_label" + args.suffix
    else:
        fname = "hand_preprocessed_untrimmed_label" + args.suffix
    dump_dataset("../data/%s.csv" % fname, untrimmed_scaled_df_w_labels)
    print("[INFO] Done: dump_untrimmed_label")

if args.dump_untrimmed_no_label:
    untrimmed_scaled_df_wo_labels = remove_cols_from_dataframe(
        untrimmed_scaled_df_w_labels, CLASS_COL_NAME
    )
    if not args.precomputed_stats:
        fname = "untrimmed_no_label" + args.suffix
    else:
        fname = "hand_preprocessed_untrimmed_no_label" + args.suffix
    dump_dataset("../data/%s.csv" % fname, untrimmed_scaled_df_wo_labels)
    print("[INFO] Done: dump_untrimmed_no_label")

if args.dump_trimmed_label:
    trimmed_scaled_df_w_labels = remove_cols_from_dataframe(
        untrimmed_scaled_df_w_labels, UNUSED_COL_NAMES
    )
    if not args.precomputed_stats:
        fname = "trimmed_target_label" + args.suffix
    else:
        fname = "hand_preprocessed_trimmed_label" + args.suffix
    dump_dataset("../data/%s.csv" % fname, trimmed_scaled_df_w_labels)
    print("[INFO] Done: dump_trimmed_label")

if args.dump_trimmed_no_label:
    trimmed_scaled_df_wo_labels = remove_cols_from_dataframe(
        untrimmed_scaled_df_w_labels, UNUSED_COL_NAMES_CLASS
    )
    if not args.precomputed_stats:
        fname = "trimmed_target_no_label" + args.suffix
    else:
        fname = "hand_preprocessed_trimmed_no_label" + args.suffix
    dump_dataset("../data/%s.csv" % fname, trimmed_scaled_df_wo_labels)
    print("[INFO] Done: dump_trimmed_no_label")
