#!/usr/bin/env python
# coding: utf-8

# @Shitong Zhu, SRA 2019
# This script simply shuffle and split the dataset into train and test
# sets

from sklearn.utils import shuffle
import math

import argparse

BASE_DATA_DIR = "../data/"

parser = argparse.ArgumentParser(
    description='Parse arguments (see description in source file)')
parser.add_argument('--test-set-size', type=int)
parser.add_argument('--trimmed-w-label', type=str)
parser.add_argument('--trimmed-wo-label', type=str)
parser.add_argument('--untrimmed-w-label', type=str)
parser.add_argument('--untrimmed-wo-label', type=str)
parser.add_argument('--augment-factor', type=int, default=1)

args = parser.parse_args()

trimmed_w_label = BASE_DATA_DIR + args.trimmed_w_label
trimmed_wo_label = BASE_DATA_DIR + args.trimmed_wo_label
untrimmed_w_label = BASE_DATA_DIR + args.untrimmed_w_label
untrimmed_wo_label = BASE_DATA_DIR + args.untrimmed_wo_label

AD_IDX = []
AD_COUNT, NONAD_COUNT = 0, 0


def read_dataset(fname):
    with open(fname, 'r') as fin:
        data = fin.readlines()
    header = data[0]
    del data[0]
    return header, data


def augment_dataset(dataset, record=False):
    global AD_IDX, AD_COUNT, NONAD_COUNT
    print("Size of dataset before augmentation: %d" % len(dataset))
    if record:
        for i in range(len(dataset)):
            label = dataset[i].strip().split(',')[-1]
            if label in {'1', "AD"}:
                AD_IDX.append(i)
                AD_COUNT += 1
            else:
                NONAD_COUNT += 1
        for i in range(math.ceil(float(NONAD_COUNT) / AD_COUNT)):
            AD_IDX.extend(AD_IDX)
    print("Size of augmentation: %d; AD count: %d; NONAD count: %d" %
          (len(AD_IDX), AD_COUNT, NONAD_COUNT))
    for i in range(NONAD_COUNT * args.augment_factor - AD_COUNT):
        dataset.append(dataset[AD_IDX[i]])
    print("Size of dataset after augmentation: %d" % len(dataset))
    return dataset


def partition(dataset, fname, header, test_set_size, exlude_target_exp=False, debug=True):
    top_domain_set = set()
    diff_domain_set = set()
    if exlude_target_exp:
        TOP_N = 500
        with open("../misc/alexa_top_actual_list.csv") as fin:
            data = fin.readlines()
        cnt = 0
        for i in range(len(data)):
            cnt += 1
            if cnt > TOP_N:
                break
            rank, alexa_domain, actual_domain = data[i].strip().split(",")
            top_domain_set.add(actual_domain)

        with open(BASE_DATA_DIR + "training_dataset.csv") as fin:
            unnorm_ds = fin.readlines()
            del unnorm_ds[0]

    test_set, train_set = [], []
    new_ds = []

    for i in range(len(dataset)):
        is_target = False
        for domain in list(top_domain_set):
            if domain in unnorm_ds[i].split(',')[0]:
                diff_domain_set.add(unnorm_ds[i].split(',')[0])
                is_target = True
                break
        if is_target:
            test_set.append(dataset[i])
        else:
            new_ds.append(dataset[i])

    cnt_target = len(test_set)

    if debug:
        print("Number of target examples:", cnt_target)
        print("Number of diff domains:", len(diff_domain_set))
        print("Size of new ds:", len(new_ds))

    shuffled_dataset = shuffle(new_ds, random_state=222)

    train_set = shuffled_dataset[:-(test_set_size-cnt_target)]
    test_set += shuffled_dataset[-(test_set_size-cnt_target):]

    if debug:
        print("Sizes:", len(train_set), len(test_set))
    return train_set, test_set


def dump_dataset(dataset, fname, suffix, header, debug=True):
    fname = fname.split('.csv')[0] + suffix + '.csv'
    if debug:
        print("Filename:", fname)
    with open(fname, 'w') as fout:
        fout.writelines([header] + dataset)


header, ds1 = read_dataset(trimmed_w_label)
_, ds2 = read_dataset(trimmed_wo_label)
_, ds3 = read_dataset(untrimmed_w_label)
_, ds4 = read_dataset(untrimmed_wo_label)

ds1_train, ds1_test = partition(
    ds1, trimmed_w_label, header, args.test_set_size)
ds2_train, ds2_test = partition(
    ds2, trimmed_wo_label, header, args.test_set_size)
ds3_train, ds3_test = partition(
    ds3, untrimmed_w_label, header, args.test_set_size)
ds4_train, ds4_test = partition(
    ds4, untrimmed_wo_label, header, args.test_set_size)

dump_dataset(ds1_train, trimmed_w_label, "_train_set", header)
dump_dataset(ds2_train, trimmed_wo_label, "_train_set", header)
dump_dataset(ds3_train, untrimmed_w_label, "_train_set", header)
dump_dataset(ds4_train, untrimmed_wo_label, "_train_set", header)

ds3_train_aug = augment_dataset(ds3_train, record=True)
ds1_train_aug = augment_dataset(ds1_train)
ds2_train_aug = augment_dataset(ds2_train)
ds4_train_aug = augment_dataset(ds4_train)

dump_dataset(ds1_train_aug, trimmed_w_label, "_augmented_train_set", header)
dump_dataset(ds2_train_aug, trimmed_wo_label, "_augmented_train_set", header)
dump_dataset(ds3_train_aug, untrimmed_w_label, "_augmented_train_set", header)
dump_dataset(ds4_train_aug, untrimmed_wo_label, "_augmented_train_set", header)

dump_dataset(ds1_test, trimmed_w_label, "_test_set", header)
dump_dataset(ds2_test, trimmed_wo_label, "_test_set", header)
dump_dataset(ds3_test, untrimmed_w_label, "_test_set", header)
dump_dataset(ds4_test, untrimmed_wo_label, "_test_set", header)

AD_IDX = []
AD_COUNT, NONAD_COUNT = 0, 0

ds3_test_aug = augment_dataset(ds3_test, record=True)
ds1_test_aug = augment_dataset(ds1_test)
ds2_test_aug = augment_dataset(ds2_test)
ds4_test_aug = augment_dataset(ds4_test)

dump_dataset(ds1_test_aug, trimmed_w_label, "_augmented_test_set", header)
dump_dataset(ds2_test_aug, trimmed_wo_label, "_augmented_test_set", header)
dump_dataset(ds3_test_aug, untrimmed_w_label, "_augmented_test_set", header)
dump_dataset(ds4_test_aug, untrimmed_wo_label, "_augmented_test_set", header)
