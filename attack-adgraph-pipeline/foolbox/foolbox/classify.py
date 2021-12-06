#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms  # import cross_val_scores
import csv
import os
import pickle

import argparse

DATA_DIR = '../data/'

CATEGORICAL_FEATURE_NAME_TO_IDX = {
    "FEATURE_NODE_CATEGORY": 13,
    "FEATURE_FIRST_PARENT_TAG_NAME": 23,
    "FEATURE_FIRST_PARENT_SIBLING_TAG_NAME": 26,
    "FEATURE_SECOND_PARENT_TAG_NAME": 42,
    "FEATURE_SECOND_PARENT_SIBLING_TAG_NAME": 45,
}

REVERSED_LABLE = {'1': "AD", '0': "NONAD"}

events, tag_1, tag_2, tag_3, tag_4 = [], [], [], [], []


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


def setup_clf(pickle_path):
    with open(pickle_path, 'rb') as fin:
        clf = pickle.load(fin)
    return clf


def setup_one_hot(events_in, tag_1_in, tag_2_in, tag_3_in, tag_4_in):
    global events, tag_1, tag_2, tag_3, tag_4
    events = events_in
    tag_1 = tag_1_in
    tag_2 = tag_2_in
    tag_3 = tag_3_in
    tag_4 = tag_4_in


def transform_row(row):
    global events, tag_1, tag_2, tag_3, tag_4
    row[13] = events.index(row[13])

    if row[23] in tag_1:
        row[23] = tag_1.index(row[23])
    elif row[23].strip() == '':
        row[23] = 0
    else:
        row[23] = 1

    if row[26] in tag_2:
        row[26] = tag_2.index(row[26])
    elif row[26].strip() == '':
        row[26] = 0
    else:
        row[26] = 1

    if row[42] in tag_3:
        row[42] = tag_3.index(row[42])
    elif row[42].strip() == '':
        row[42] = 0
    else:
        row[42] = 1

    if row[45] in tag_4:
        row[45] = tag_4.index(row[45])
    elif row[45].strip() == '':
        row[45] = 0
    else:
        row[45] = 1

    row[4] = round(float(row[4]), 3)
    row[5] = round(float(row[5]), 3)
    row[10] = round(float(row[10]), 3)
    row[32] = round(float(row[32]), 3)
    row[51] = round(float(row[51]), 3)

    return row


def predict(x, clf):
    TO_EXCLUDE = {0, 1, 9, 31, 50}
    if len(x) == 65:
        x_new = []
        cnt = 0
        for i in range(70):
            if i in TO_EXCLUDE:
                x_new.append("dummy")
            else:
                x_new.append(x[cnt])
                cnt += 1
        x = x_new
    transformed_x = transform_row(x)
    # [:-1] to remove class
    # print(transformed_x)
    trimmed_x = [element for i, element in enumerate(
        transformed_x) if i not in TO_EXCLUDE]
    trimmed_x = np.array([trimmed_x])
    res = clf.predict(trimmed_x)
    return [REVERSED_LABLE[res[0]]]
