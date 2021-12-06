#!/usr/bin/env python
# coding: utf-8

# @Shitong Zhu, SRA 2019
# This script appends the classification results from Weka model
# to the dataset file

# Required args:
# --original-ds: str, specifies the dataset file that has not been preprocessed
# (it can be either unprocessed or processed)
# --model-output: str, the model output file from Weka
# --appended-label-file: str, path for appended output w/ labels
# --appended-preds-file: str, path for appended output w/ prediction scores

import copy
import argparse

LABEL_MAP = {'1': '1', '0': '0'}
BASE_DATA_DIR = "../data/"

parser = argparse.ArgumentParser(
    description='Parse arguments (see description in source file)')
parser.add_argument('--original-ds', type=str)
parser.add_argument('--model-output', type=str)
parser.add_argument('--appended-label-file', type=str)
args = parser.parse_args()


def read_dataset(file):
    with open(file, 'r') as fin:
        data = fin.readlines()
        data = list(map(lambda l: l.strip(), data))
    header = data[0]
    del data[0]
    return header, data


def read_label_pred(file):
    model_labels = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        data = list(map(lambda l: l.strip(), data))
    for r in data:
        model_label = LABEL_MAP[r]
        model_labels.append(model_label)
    return model_labels


def append_to_dataset(header, dataset, suffix):
    dataset_copy = copy.deepcopy(dataset)
    assert len(dataset) == len(suffix), "[ERROR] Different sizes!"
    for i in range(len(dataset)):
        x_str_wo_label = ','.join(dataset[i].split(',')[:-1])
        dataset_copy[i] = x_str_wo_label + ',' + suffix[i] + '\n'
    return [header + '\n'] + dataset_copy


def dump_new_dataset(fname, new_dataset):
    with open(fname, 'w') as fout:
        fout.writelines(new_dataset)


# "unnormalized_one_hot_encoded_target.csv"
# Read header and original dataset
header, dataset = read_dataset(BASE_DATA_DIR + args.original_ds)

# Read preds and labels from model output file
labels = read_label_pred(BASE_DATA_DIR + args.model_output)
dataset_with_labels = append_to_dataset(header, dataset, labels)

# unnormalized_target_w_unused_numbered_class_gt_preds.csv
# "data/unnormalized_target_w_unused_numbered_class_gt.csv"
# Dump appended datasets
dump_new_dataset(BASE_DATA_DIR + args.appended_label_file, dataset_with_labels)
