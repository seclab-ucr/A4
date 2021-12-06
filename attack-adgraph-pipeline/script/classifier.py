#!/usr/bin/env python
# -*- coding: utf-8 -*-

from generate_all_def import read_one_hot_feature_list

import pickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms  # import cross_val_scores
import csv
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
import os
import argparse

DATA_DIR = '../data/'

parser = argparse.ArgumentParser(
    description='Parse arguments (see description in source file)')
parser.add_argument('--dataset-fpath', default='dataset_1111.csv', type=str)
parser.add_argument('--dump-mode', action='store_true')
parser.add_argument('--suffix', type=str)
args = parser.parse_args()

DUMP_MODE = args.dump_mode
COMPUTE_CV = True

data_file = DATA_DIR + args.dataset_fpath
one_hot_feature_list = read_one_hot_feature_list(data_file)

events = sorted(list(one_hot_feature_list["FEATURE_NODE_CATEGORY"]))

tag_1 = sorted(list(one_hot_feature_list["FEATURE_FIRST_PARENT_TAG_NAME"]))
tag_2 = sorted(
    list(one_hot_feature_list["FEATURE_FIRST_PARENT_SIBLING_TAG_NAME"]))
tag_3 = sorted(list(one_hot_feature_list["FEATURE_SECOND_PARENT_TAG_NAME"]))
tag_4 = sorted(
    list(one_hot_feature_list["FEATURE_SECOND_PARENT_SIBLING_TAG_NAME"]))


def transform_row(row):
    global tag_1, tag_2, tag_3, tag_4, events

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


def cv_confusion_matrix(clf, X, y, folds=10):
    skf = StratifiedKFold(n_splits=folds)
    cv_iter = skf.split(X, y)
    cms = []
    for train, test in cv_iter:
        clf.fit(X[train], y[train])
        res = clf.predict(X[test])
        cm = confusion_matrix(y[test], res, labels=clf.classes_)
        cms.append(cm)
    print(clf.classes_)
    return np.sum(np.array(cms), axis=0)


to_exclude = {0, 1, 9, 31, 50}
TestFileCsvReader = csv.reader(open(data_file, 'r'), delimiter=',')
testdata = []
labels = []
TestdataIds = []
next(TestFileCsvReader)
for row in TestFileCsvReader:
    row = transform_row(row)
    d = [element for i, element in enumerate(row[:-1]) if i not in to_exclude]
    testdata.append(np.array(d))
    labels.append(row[-1])

print("[INFO] Size of dataset: %d" % len(testdata))
if DUMP_MODE:
    # n_estimators is numTree. max_features is numFeatures
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=1, criterion="entropy")
    clf.fit(np.asarray(testdata), np.asarray(labels))


if DUMP_MODE:
    with open("../model/rf.pkl", 'wb') as fout:
        pickle.dump(clf, fout)
else:
    with open("../model/rf.pkl", 'rb') as fin:
        clf = pickle.load(fin)

if not DUMP_MODE:
    clf_res = clf.predict(np.asarray(testdata))
    clf_res = list(map(lambda e: str(e) + "\n", clf_res))

    with open(DATA_DIR + "%s_model_output.txt" % args.suffix, 'w') as fout:
        fout.writelines(clf_res)
    print("[INFO] Output produced and dumped.")

    if COMPUTE_CV:
        scores = cv_confusion_matrix(
            clf, np.asarray(testdata), np.asarray(labels), 10
        )
        print(scores)
        tp = scores[0][0]
        tn = scores[1][1]
        fp = scores[1][0]
        fn = scores[0][1]

        accuracy = round(((tp+tn)*1.0/(tp+tn+fp+fn) * 1.0)*100, 2)
        FPR = round(((fp)*1.0/(tn+fp)*1.0)*100, 2)
        Recall = round(((tp)*1.0/(tp+fn)*1.0)*100, 2)
        precesion = round(((tp)*1.0/(tp+fp)*1.0)*100, 2)

        print("[INFO] ACCURACY:" + str(accuracy))
        print("[INFO] FPR:" + str(FPR))
        print("[INFO] Recall:" + str(Recall))
        print("[INFO] Precision:" + str(precesion))
