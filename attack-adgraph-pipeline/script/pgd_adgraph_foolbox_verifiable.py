#!/usr/bin/env python
# coding: utf-8

# @Shitong Zhu, SRA 2019
# This script generates adversarial examples using PGD that are actionable
# (meaning they can be mapped back to the raw input space and still remain
# adversarial)

import time
import os
import sys
import argparse
import math
import logging
from tensorflow import keras
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import Counter
import numpy as np
import copy

CWD = os.getcwd()
FOOLBOX_LIB_DIR = "/".join(CWD.split("/")[:-1]) + "/foolbox"
sys.path.append(FOOLBOX_LIB_DIR)
import foolbox  # noqa
from foolbox.models import KerasModel  # noqa
from foolbox.attacks import ProjectedGradientDescentAttack  # noqa
from foolbox.criteria import TargetClass  # noqa


epoch_time = str(int(time.time() * 10000000))

BASE_DATA_DIR = "../data/"
BASE_DEF_DIR = "../def/"
BASE_RES_DIR = "../res/"
BASE_REPORT_DIR = "../report/"

parser = argparse.ArgumentParser(
    description='Generatete adversarial examples (see description in source file)')
parser.add_argument('--target-file', type=str)
parser.add_argument('--unnorm-target-file', type=str)
parser.add_argument('--target-gt-file', type=str)
parser.add_argument('--model-file', type=str)
parser.add_argument('--feature-defs', type=str)
parser.add_argument('--feature-idx', type=str)
parser.add_argument('--unnorm-feature-idx', type=str)
parser.add_argument('--target-set-size', type=int, default=100000)
parser.add_argument('--start-idx', type=int, default=0)
parser.add_argument('--end-idx', type=int, default=100000)
parser.add_argument('--browser-id', type=int)
parser.add_argument('--preprocess-feature-defs', type=str)
parser.add_argument('--augment-pgd', dest='augment_pgd', action='store_true')
parser.add_argument('--feature-set', type=str)

args = parser.parse_args()


def read_dataset(file, model, verbal=False):
    def get_label(x, model, remove_idx_set):
        x_trimmed = []

        for i in range(len(x)):
            if i not in remove_idx_set:
                x_trimmed.append(x[i])

        x_trimmed_np = np.array([x_trimmed])
        preds = model.predict(x_trimmed_np)[0]

        if preds[0] > preds[1]:
            return False
        else:
            return True

    def split_by_pred(x_lst, model, remove_idx_set, cutoff=1000):
        ad, nonad = [], []

        for i in range(len(x_lst)):
            if i > cutoff:
                break
            if get_label(x_lst[i].split(','), model, remove_idx_set):
                if verbal:
                    print("[INFO] #%d data point: AD" % i)
                ad.append(x_lst[i])
            else:
                if verbal:
                    print("[INFO] #%d data point: NONAD" % i)
                nonad.append(x_lst[i])

        return ad, nonad

    _IDX_TO_REMOVE = {0, 1, 9, 198, 297}

    data_lst = []
    domain_set = set()

    with open(file, 'r') as fin:
        data = fin.readlines()

    # remove headers
    del data[0]

    for r in data:
        r = r.strip()
        # handles missing values
        r = r.replace('?', '0')
        if r and not r.startswith('DOMAIN_NAME'):
            data_lst.append(r)
            domain_set.add(r.split(',')[0])

    ad, nonad = split_by_pred(data_lst, model, _IDX_TO_REMOVE, len(data_lst))
    return ad, nonad, list(domain_set)


def read_gt_dataset(file, verbal=False):
    def split_by_pred(x_lst, label_dict, cutoff=1000):
        ad, nonad = [], []
        for i in range(len(x_lst)):
            if i > cutoff:
                break
            if x_lst[i].split(',')[-1] == label_dict['AD']:
                if verbal:
                    print("[INFO][GT] #%d data point: AD" % i)
                unlabeled_x = ','.join(x_lst[i].split(',')[:-1])
                ad.append(unlabeled_x)
            elif x_lst[i].split(',')[-1] == label_dict['NONAD']:
                if verbal:
                    print("[INFO][GT] #%d data point: NONAD" % i)
                unlabeled_x = ','.join(x_lst[i].split(',')[:-1])
                nonad.append(unlabeled_x)
        return ad, nonad

    _LABEL_DICT = {'AD': '1', 'NONAD': '0'}
    data_lst = []

    with open(file, 'r') as fin:
        data = fin.readlines()

    del data[0]

    for r in data:
        r = r.strip()
        # handles missing values
        r = r.replace('?', '0')
        data_lst.append(r)

    ad, nonad = split_by_pred(data_lst, _LABEL_DICT, len(data_lst))

    return ad, nonad


def dump_a_list(lst, file):
    lst = list(map(lambda l: l+"\n", lst))
    with open(file, 'w') as fout:
        fout.writelines(lst)


def read_a_file(file):
    with open(file, 'r') as fin:
        data = fin.readlines()
    data = list(map(lambda l: l.strip(), data))
    return data


def check_is_file(file):
    import os.path
    if os.path.isfile(file):
        return True
    else:
        return False


def check_is_new_file(fname):
    import os.path
    fname_mtime = os.path.getmtime(fname)
    if (int(epoch_time) - int(fname_mtime)) / 3600 / 24 > 7:
        return False
    else:
        return True


def get_partitioned_dataset(ds_dir, model, timestamp, debug=True):
    _MASTER_FILE = args.target_file
    _AD_FILE = "hand_preprocessed_target_no_label_ad_%s.csv" % timestamp
    _NONAD_FILE = "hand_preprocessed_target_no_label_nonad_%s.csv" % timestamp

    master_fname = ds_dir + _MASTER_FILE
    ad_fname = ds_dir + _AD_FILE
    nonad_fname = ds_dir + _NONAD_FILE

    files_exist = False

    if check_is_file(ad_fname) and check_is_file(nonad_fname):
        if check_is_new_file(ad_fname) and check_is_new_file(nonad_fname):
            files_exist = True
        else:
            files_exist = False

    if files_exist:
        ad = read_a_file(ad_fname)
        nonad = read_a_file(nonad_fname)
    else:
        ad, nonad, domain_lst = read_dataset(master_fname, model)
        dump_a_list(ad, ad_fname)
        dump_a_list(nonad, nonad_fname)

    if debug:
        print("Filenames:")
        print(files_exist)
        print(master_fname)
        print(ad_fname)
        print(nonad_fname)

    return ad, nonad


def get_partitioned_gt_dataset(ds_dir, timestamp, debug=True):
    _MASTER_FILE = args.target_gt_file
    _AD_FILE = "hand_preprocessed_target_no_label_gt_ad_%s.csv" % timestamp
    _NONAD_FILE = "hand_preprocessed_target_no_label_gt_nonad_%s.csv" % timestamp

    master_fname = ds_dir + _MASTER_FILE
    ad_fname = ds_dir + _AD_FILE
    nonad_fname = ds_dir + _NONAD_FILE

    files_exist = False

    if check_is_file(ad_fname) and check_is_file(nonad_fname):
        if check_is_new_file(ad_fname) and check_is_new_file(nonad_fname):
            files_exist = True
        else:
            files_exist = False

    if files_exist:
        ad = read_a_file(ad_fname)
        nonad = read_a_file(nonad_fname)
    else:
        ad, nonad = read_gt_dataset(master_fname)
        dump_a_list(ad, ad_fname)
        dump_a_list(nonad, nonad_fname)

    if debug:
        print("Filenames:")
        print(files_exist)
        print(master_fname)
        print(ad_fname)
        print(nonad_fname)

    return ad, nonad


def read_preprocessing_feature_defs(file):
    defs = {}
    with open(file, 'r') as fin:
        data = fin.readlines()
    for i in range(len(data)):
        r = data[i].strip()
        if len(r.split(',')) == 3:
            [ftype, maxn, minn] = r.split(',')
            val = [maxn, minn]
        if len(r.split(',')) == 2:
            [ftype, val] = r.split(',')
        if len(r.split(',')) == 1:
            [ftype] = r.split(',')
            val = None
        defs[i] = {'type': ftype, 'val': val}
    return defs


def post_process_adv_x(adv_x_lst, feature_types):
    # handles fields to ensure the validity of generated examples
    processed_adv_x_lst = []

    for i in range(len(adv_x_lst)):
        adv_x = np.copy(adv_x_lst[i])
        processed_adv_x = np.copy(adv_x_lst[i])

        for i in range(len(adv_x)):
            adv_val = adv_x[i]
            if feature_types[i] == 'f':
                processed_adv_val = adv_val
                processed_adv_val = max(processed_adv_val, 0.0)
                processed_adv_val = min(processed_adv_val, 1.0)
            if feature_types[i] == 'b':
                if abs(adv_val - 1.0) > abs(adv_val - 0.0):
                    processed_adv_val = 0
                else:
                    processed_adv_val = 1

            processed_adv_x[i] = processed_adv_val

        lookahead_cnt = 0
        for i in range(len(adv_x)):
            if lookahead_cnt - 1 > 0:
                lookahead_cnt -= 1
                continue
            if feature_types[i] == 'c':
                j = i
                while feature_types[j] == 'c' and j + 1 < len(adv_x):
                    j += 1
                categorical_interval_end = j
                maxn = -10000
                maxn_idx = i
                for j in range(i, categorical_interval_end):
                    if adv_x[j] > maxn:
                        maxn = adv_x[j]
                        maxn_idx = j
                for j in range(i, categorical_interval_end):
                    if j == maxn_idx:
                        processed_adv_val = 1
                    else:
                        processed_adv_val = 0
                    processed_adv_x[j] = processed_adv_val
                    lookahead_cnt += 1

        processed_adv_x_lst.append(processed_adv_x)

    return processed_adv_x_lst


def read_feature_defs(file):
    with open(file, 'r') as fin:
        data = fin.readlines()
    defs = map(lambda l: l.strip(), data)
    return list(defs)


def generate_np(attack,
                ad_x,
                perturbe_ad,
                verbal=False,
                verify=False,
                model=None,
                browser_id=None,
                process=False,
                feature_defs=None,
                perturbable_idx_set=None,
                only_increase_idx_set=None,
                normalization_ratios=None,
                logger=None,
                attack_logger=None,
                iterations=100,
                stepsize=0.05,
                epsilon=0.3,
                enforce_interval=1,
                debug=False,
                element_wise_diff=True,
                print_x=False,
                map_back_mode=False,
                feature_idx_map=None):

    def trim_x(x, remove_idx_set):
        x_trimmed = []
        for i in range(len(x)):
            if i not in remove_idx_set:
                x_trimmed.append(float(x[i]))
        return np.array(x_trimmed)

    def generate_trimmed_x_str(x, remove_idx_set):
        x_trimmed = []
        for i in range(len(x)):
            if i not in remove_idx_set:
                x_trimmed.append(x[i])
        return ','.join(x_trimmed)

    def get_label(x, model, verbal=False):
        x_np = np.array([x])
        preds = model.predict(x_np)[0]
        if verbal:
            print(preds)
        if preds[0] > preds[1]:
            return False
        else:
            return True

    def get_preds(x, model):
        x_np = np.array([x])
        preds = model.predict(x_np)[0]
        return preds

    def untrim_x(original, trimmed, remove_idx_set):
        x_copy = np.copy(original).tolist()
        cnt = 0
        for i in range(len(original)):
            if i not in remove_idx_set:
                x_copy[i] = trimmed[cnt]
                cnt += 1
        return x_copy

    _IDX_TO_REMOVE = {0, 1, 9, 198, 297}

    ad_x = ad_x.split(',')
    # pass the ID to foolbox
    request_id = ','.join([ad_x[0], ad_x[1]])

    _ad_x = trim_x(ad_x, _IDX_TO_REMOVE)

    if perturbe_ad:
        label = 1
    else:
        label = 0

    adv_x = attack(input_or_adv=_ad_x,
                   label=label,
                   iterations=iterations,
                   stepsize=stepsize,
                   epsilon=epsilon,
                   unpack=True,
                   binary_search=False,
                   random_start=False,
                   return_early=True,  # since we can now verify on remote model
                   perturbable_idx_set=perturbable_idx_set,
                   only_increase_idx_set=only_increase_idx_set,
                   feature_defs=feature_defs,
                   normalization_ratios=normalization_ratios,
                   enforce_interval=enforce_interval,
                   request_id=request_id,
                   model=model,
                   browser_id=browser_id,
                   logger=attack_logger,
                   map_back_mode=map_back_mode,
                   feature_idx_map=feature_idx_map)
    if adv_x is None:
        print("[INFO] Failed finding the adv_x")
        return None, None, None

    if process:
        adv_x = post_process_adv_x(np.array([adv_x]), feature_types)[0]

    _adv_x = untrim_x(ad_x, adv_x, _IDX_TO_REMOVE)

    if verbal:
        if print_x:
            print("[INFO] Original x:", _ad_x)
            print("[INFO] Perturbed x:", adv_x)
        if element_wise_diff:
            for i in range(len(_ad_x)):
                if _ad_x[i] != adv_x[i]:
                    print("i:", i, "Ori:", _ad_x[i], " Pert:", adv_x[i])

    if verify:
        if verbal:
            if logger:
                logger.info("preds for original: " +
                            str(get_preds(_ad_x, model)))
                logger.info("preds for perturbed: " +
                            str(get_preds(adv_x, model)))
            print("[DEBUG] Preds of original x:", get_preds(_ad_x, model))
            print("[DEBUG] Preds of perturbed x:", get_preds(adv_x, model))

    if get_label(adv_x, model) != get_label(_ad_x, model):
        if verbal:
            if logger:
                logger.info("Success!")
            print("[INFO] Success!")
        return _adv_x, ad_x, True
    else:
        if verbal:
            if logger:
                logger.info("Fail!")
            print("[INFO] Fail!")
        return _adv_x, ad_x, False


def deprocess_dataset(dataset, feature_types, verbal=False):
    def deprocess_float_feature(val, ratio, ffloat=True):
        val = float(val)
        maxn = float(ratio[0])
        minn = float(ratio[1])
        if ffloat:
            return round(float(val * (maxn - minn) + minn), 6)
        else:
            return round(val * (maxn - minn) + minn)

    def deprocess_nominal_feature(val, category_name):
        val = float(val)
        if val == 1.0:
            return category_name
        elif val == 0.0:
            return None
        else:
            print("[ERROR] WTF? val: %s %s" % (str(val), category_name))
            raise Exception

    def deprocess_shift_feature(val, offset):
        val = float(val)
        offset = float(offset)
        return int(math.ceil(val + offset))

    deprocessed_dataset = []

    for i in range(len(dataset)):
        features = dataset[i]
        deprocessed_features = []

        for j in range(len(feature_types)):
            assert feature_types[j]['type'] in FEATURE_TYPES, "[ERROR] Feature type not supported!"

            if feature_types[j]['type'] == 'F':
                ratio = feature_types[j]['val']
                deprocessed_features.append(
                    deprocess_float_feature(features[j], ratio, ffloat=False))

            if feature_types[j]['type'] == 'FF':
                ratio = feature_types[j]['val']
                deprocessed_features.append(
                    deprocess_float_feature(features[j], ratio, ffloat=True))

            if feature_types[j]['type'] == 'C':
                category_name = feature_types[j]['val']
                new_val = deprocess_nominal_feature(features[j], category_name)
                if new_val is not None:
                    deprocessed_features.append(new_val)

            if feature_types[j]['type'] == 'S':
                offset = feature_types[j]['val']
                deprocessed_features.append(
                    deprocess_shift_feature(features[j], offset))

            if feature_types[j]['type'] == 'B':
                val = features[j]
                deprocessed_features.append(int(float(val)))

            if feature_types[j]['type'] == 'D':
                val = features[j]
                deprocessed_features.append(val)

            # label column
            if feature_types[j]['type'] == 'L':
                label = features[j]
                deprocessed_features.append(label)

        deprocessed_features = map(lambda l: str(l), deprocessed_features)
        features_str = ','.join(deprocessed_features) + '\n'
        deprocessed_dataset.append(features_str)

    return deprocessed_dataset


def read_unnormalized_dataset(fname):
    def check_has_missing_value(example, tag_feature_idx_set):
        example_lst = example.split(',')
        for i in range(len(example_lst)):
            if i in tag_feature_idx_set and example_lst[i] == "?":
                return True
        return False

    ds_dict = {}
    examples_w_missing_value_set = set()
    with open(fname, 'r') as fin:
        data = fin.readlines()
        del data[0]
    for l in data:
        domain = l.split(',')[0]
        url_id = l.split(',')[1]
        request_id = domain + url_id
        l_processed = l.strip()
        l_processed = ','.join(l_processed.split(',')[:-1])
        if check_has_missing_value(l_processed, TAG_FEATURE_IDX):
            examples_w_missing_value_set.add(request_id)
        ds_dict[request_id] = l_processed
    return ds_dict, examples_w_missing_value_set


def check_if_in_missing_value_set(example, examples_w_missing_value_set):
    domain = example.split(',')[0]
    url_id = example.split(',')[1]
    request_id = domain + url_id
    if request_id in examples_w_missing_value_set:
        return True
    else:
        return False


def read_feature_idx(fname):
    feature_idx_map = {}
    with open(fname, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        idx, feature_name = r.split(',')
        feature_idx_map[feature_name] = int(idx)
    return feature_idx_map


def is_numeric(string):
    try:
        num = float(string)
    except Exception:
        return False
    return True


def read_unnormlized_feature_idx(fname):
    unnormalized_feature_idx_dict = {}
    with open(fname, 'r') as fin:
        data = fin.readlines()
    for r in data:
        r = r.strip()
        idx, feature_name = r.split(',')
        unnormalized_feature_idx_dict[int(idx)] = feature_name
    return unnormalized_feature_idx_dict


def compare_two_x(x_1,
                  x_2,
                  dummy_feature_idx_set=None,
                  float_feature_idx_set=None,
                  integer_feature_idx_set=None,
                  tag_feature_idx_set=None,
                  return_final=False,
                  return_diff=False,
                  feature_idx_map=None):
    def is_numerically_indifferent(x_1, x_2):
        x_1 = float(x_1)
        x_2 = float(x_2)
        if abs(x_1 - x_2) / (x_1 + x_2) < FLOAT_INDIFFERENCE_THRES:
            return True
        else:
            return False

    assert len(x_1) == len(
        x_2), "Unequel size of two Xs: %d vs %d" % (len(x_1), len(x_2))

    if return_final:
        final_ret = copy.copy(x_1)
    if return_diff:
        domain = x_1[0]
        url_id = x_1[1]
        diff_str = domain + ',' + url_id + "\n"

    for i in range(len(x_1)):
        if dummy_feature_idx_set and i in dummy_feature_idx_set:
            continue
        if float_feature_idx_set and i in float_feature_idx_set:
            if is_numerically_indifferent(x_1[i], x_2[i]):
                continue
        if is_numeric(x_1[i]) and is_numeric(x_2[i]):
            if float(x_1[i]) == float(x_2[i]):
                continue
        if integer_feature_idx_set and i in integer_feature_idx_set:
            if abs(int(x_1[i]) - int(x_2[i])) <= INTEGER_INDIFFERENCE_THRES:
                continue
        if tag_feature_idx_set and i in tag_feature_idx_set:
            if x_1[i] == '?':
                continue
        if x_1[i] != x_2[i]:
            print(
                "%s" % (UNNORMALIZED_FEATURE_IDX[i]), "original:", x_1[i], "perturbed:", x_2[i])
            if return_final:
                final_ret[i] = x_2[i]
            if return_diff:
                if i not in AUTOMATIC_COMPUTED_FEATURES:
                    diff = int(x_2[i]) - int(x_1[i])
                    new_diff_str = feature_idx_map[i] + ',' + str(diff) + '\n'
                    diff_str += new_diff_str

    ret = {}
    if return_final:
        ret['x'] = ','.join(final_ret)
    if return_diff:
        ret['diff'] = diff_str
    if ret != []:
        return ret


def _get_logger(mode):
    logger = logging.getLogger("attack-logger")
    formatter = logging.Formatter('%(message)s - %(asctime)s')
    logger.setLevel(logging.INFO)
    if mode == "aug-pgd":
        fileHandler = logging.FileHandler(
            '../report/aug_pgd_attack.log.%s' % epoch_time, 'w')
    if mode == "original-pgd":
        fileHandler = logging.FileHandler(
            '../report/original_pgd_attack.%s' % epoch_time, 'w')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger


MODEL_FILE = args.model_file
TIMESTAMP = '_'.join(MODEL_FILE.split(".")[0].split("_")[2:])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fileHandler = logging.FileHandler(
    '../report/experiment_%s.log' % TIMESTAMP, mode='w')
logger.addHandler(fileHandler)

NUM_FEATURES = 312

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(NUM_FEATURES,)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights("../model/%s" % MODEL_FILE)

ad, nonad, = get_partitioned_dataset("../data/", model, TIMESTAMP)
gt_ad, gt_nonad = get_partitioned_gt_dataset("../data/", TIMESTAMP)

gt_ad_set, gt_nonad_set = set(gt_ad), set(gt_nonad)

print("Stats/counts for dataset used:")
print(len(ad), len(nonad))
print(len(gt_ad), len(gt_nonad))

test_x_lst = []

PERTURBE_AD = True
USE_ALL = True
CUT_OFF = args.target_set_size
DOMAIN_NAME = {"https://www.washingtonpost.com/"}

TAG_FEATURE_IDX = {23, 26, 42, 45}

unnormalized_dataset, examples_w_missing_value_set = read_unnormalized_dataset(
    "../data/%s" % args.unnorm_target_file)

FILTER_MISSING_VALUES = False

if not FILTER_MISSING_VALUES:
    examples_w_missing_value_set = set()

cnt = 0
diff_cnt = 0

if PERTURBE_AD:
    for i in range(len(ad)):
        if cnt > CUT_OFF:
            break
        if check_if_in_missing_value_set(ad[i], examples_w_missing_value_set):
            continue
        if USE_ALL:
            if ad[i] in gt_ad_set:
                test_x_lst.append(ad[i])
                cnt += 1
            else:
                diff_cnt += 1
                # print("[INFO] AD in remote model is not AD locally: %d" % diff_cnt)
        elif ad[i].split(',')[0] in DOMAIN_NAME:
            if ad[i] in gt_ad_set:
                test_x_lst.append(ad[i])
                cnt += 1
            else:
                diff_cnt += 1
                # print("[INFO] AD in remote model is not AD locally: %d" % diff_cnt)
else:
    for i in range(len(nonad)):
        if cnt > CUT_OFF:
            break
        if check_if_in_missing_value_set(nonad[i], examples_w_missing_value_set):
            continue
        if USE_ALL:
            if ad[i] in gt_nonad_set:
                test_x_lst.append(nonad[i])
                cnt += 1
            else:
                diff_cnt += 1
                # print(
                # "[INFO] NONAD in remote model is not NONAD locally: %d" % diff_cnt)
        elif nonad[i].split(',')[0] in DOMAIN_NAME:
            if ad[i] in gt_nonad_set:
                test_x_lst.append(nonad[i])
                cnt += 1
            else:
                diff_cnt += 1
                # print(
                # "[INFO] NONAD in remote model is not NONAD locally: %d" % diff_cnt)

print("Number of examples used as target:", len(test_x_lst))
logger.info('Inconsistent count: %d' % diff_cnt)


sess = tf.Session()
# we need this because will use generate_np() instead of
# dealying evaluations to symbolic computing graph
tf.initialize_all_variables().run(session=sess)


with tf.keras.backend.get_session().as_default():
    foolbox_model = foolbox.models.TensorFlowModel.from_keras(
        model=model, bounds=(0.0, 1.0))

pgda = ProjectedGradientDescentAttack(model=foolbox_model,
                                      criterion=TargetClass(0),
                                      distance=foolbox.distances.Linfinity)


# '../def/preprocessed_adgraph_alexa_10k_feature_defs.txt'
FEATURE_DEFS_FILE = '../def/%s' % args.feature_defs
# '../def/trimmed_wo_class_feature_idx.csv'
FEATURE_IDX_FILE = '../def/%s' % args.feature_idx

feature_types = read_feature_defs(FEATURE_DEFS_FILE)
feature_idx_map = read_feature_idx(FEATURE_IDX_FILE)

if args.feature_set == 'all':
    FNAMES_TO_PERTURBE = [
        "FEATURE_GRAPH_NODES",
        "FEATURE_GRAPH_EDGES",
        "FEATURE_GRAPH_NODES_EDGES",
        "FEATURE_GRAPH_EDGES_NODES",
        "FEATURE_URL_LENGTH",
        "FEATURE_AD_KEYWORD",
        "FEATURE_SPECIAL_CHAR_AD_KEYWORD",
        "FEATURE_SEMICOLON_PRESENT",
        "FEATURE_BASE_DOMAIN_IN_QS",
        "FEATURE_AD_DIMENSIONS_IN_QS",
        "FEATURE_INBOUND_OUTBOUND_CONNECTIONS",
        "FEATURE_ASCENDANTS_AD_KEYWORD",
        "FEATURE_SCREEN_DIMENSIONS_IN_QS",
        "FEATURE_AD_DIMENSIONS_IN_COMPLETE_URL",
        "FEATURE_FIRST_NUMBER_OF_SIBLINGS",
        "FEATURE_FIRST_PARENT_NUMBER_OF_SIBLINGS",
        "FEATURE_FIRST_PARENT_SIBLING_AD_ATTRIBUTE",
        "FEATURE_FIRST_PARENT_INBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_INBOUND_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_AVERAGE_DEGREE_CONNECTIVITY",
    ]
    FNAMES_TO_INCREASE_ONLY = [
        "FEATURE_GRAPH_NODES",
        "FEATURE_GRAPH_EDGES",
        "FEATURE_URL_LENGTH",
        "FEATURE_INBOUND_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_NUMBER_OF_SIBLINGS",
        "FEATURE_FIRST_PARENT_INBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_INBOUND_OUTBOUND_CONNECTIONS",
    ]
elif args.feature_set == 'no_url':
    FNAMES_TO_PERTURBE = [
        "FEATURE_GRAPH_NODES",
        "FEATURE_GRAPH_EDGES",
        "FEATURE_GRAPH_NODES_EDGES",
        "FEATURE_GRAPH_EDGES_NODES",
        "FEATURE_INBOUND_OUTBOUND_CONNECTIONS",
        "FEATURE_ASCENDANTS_AD_KEYWORD",
        "FEATURE_FIRST_NUMBER_OF_SIBLINGS",
        "FEATURE_FIRST_PARENT_NUMBER_OF_SIBLINGS",
        "FEATURE_FIRST_PARENT_SIBLING_AD_ATTRIBUTE",
        "FEATURE_FIRST_PARENT_INBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_INBOUND_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_AVERAGE_DEGREE_CONNECTIVITY",
    ]
    FNAMES_TO_INCREASE_ONLY = [
        "FEATURE_GRAPH_NODES",
        "FEATURE_GRAPH_EDGES",
        "FEATURE_INBOUND_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_NUMBER_OF_SIBLINGS",
        "FEATURE_FIRST_PARENT_INBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_OUTBOUND_CONNECTIONS",
        "FEATURE_FIRST_PARENT_INBOUND_OUTBOUND_CONNECTIONS",
    ]
if args.feature_set == 'only_url':
    FNAMES_TO_PERTURBE = [
        "FEATURE_URL_LENGTH",
        "FEATURE_AD_KEYWORD",
        "FEATURE_SPECIAL_CHAR_AD_KEYWORD",
        "FEATURE_SEMICOLON_PRESENT",
        "FEATURE_BASE_DOMAIN_IN_QS",
        "FEATURE_AD_DIMENSIONS_IN_QS",
        "FEATURE_SCREEN_DIMENSIONS_IN_QS",
        "FEATURE_AD_DIMENSIONS_IN_COMPLETE_URL",
    ]
    FNAMES_TO_INCREASE_ONLY = [
        "FEATURE_URL_LENGTH",
    ]


IDX_TO_PERTURBE = set()
IDX_TO_INCREASE_ONLY = set()
for f in FNAMES_TO_PERTURBE:
    IDX_TO_PERTURBE.add(feature_idx_map[f])
for f in FNAMES_TO_INCREASE_ONLY:
    IDX_TO_INCREASE_ONLY.add(feature_idx_map[f])

FEATURE_DEFS = feature_types

logger.info("IDX_TO_PERTURBE: " + str(IDX_TO_PERTURBE))
logger.info("IDX_TO_INCREASE_ONLY: " + str(IDX_TO_INCREASE_ONLY))

# '../def/hand_preprocessing_defs.csv'
PREPROCESSING_FEATURE_DEFS_FILE = BASE_DEF_DIR + args.preprocess_feature_defs

# F: float
# B: binary
# C: categorical/nominal
# S: shift (addition/subtraction)
# D: dummy (for using Weka)
FEATURE_TYPES = {'F', 'B', 'C', 'S', 'D', 'L', 'FF'}

defs = read_preprocessing_feature_defs(PREPROCESSING_FEATURE_DEFS_FILE)
defs_remote_model = read_preprocessing_feature_defs(
    BASE_DEF_DIR + "hand_preprocessing_defs_for_remote_model.csv")

succ_adv_x, succ_ori_x = [], []

ITERATIONS = 300
STEPSIZE = 0.07
EPSILON = 0.3
MAP_BACK_MODE = True

if args.augment_pgd:
    ENFORECE_INTERVAL = 15
    attack_logger = _get_logger("aug-pgd")
else:
    ENFORECE_INTERVAL = ITERATIONS - 1
    attack_logger = _get_logger("original-pgd")

DEBUG_MODE = True

logger.debug("PGD Parameters:")
logger.debug("iterations: %d" % ITERATIONS)
logger.debug("stepsize: %d" % STEPSIZE)
logger.debug("epsilon: %d" % EPSILON)
logger.debug("enforce_interval: %d" % ENFORECE_INTERVAL)

for i in range(len(test_x_lst)):
    if i < args.start_idx:
        continue
    if i > args.end_idx:
        break
    logger.info("Example #%d" % i)
    attack_logger.info("Example #%d" % i)
    print("[INFO] Generainting #%d adversarial example..." % i)
    adv_x, test_x, result = generate_np(pgda,
                                        test_x_lst[i],
                                        perturbe_ad=PERTURBE_AD,
                                        verbal=False,
                                        verify=False,
                                        model=model,
                                        browser_id=args.browser_id,
                                        process=False,
                                        feature_defs=FEATURE_DEFS,
                                        perturbable_idx_set=IDX_TO_PERTURBE,
                                        only_increase_idx_set=IDX_TO_INCREASE_ONLY,
                                        normalization_ratios=defs_remote_model,
                                        logger=logger,
                                        attack_logger=attack_logger,
                                        iterations=ITERATIONS,
                                        stepsize=STEPSIZE,
                                        epsilon=EPSILON,
                                        enforce_interval=ENFORECE_INTERVAL,
                                        debug=DEBUG_MODE,
                                        map_back_mode=MAP_BACK_MODE,
                                        feature_idx_map=feature_idx_map)
    if result:
        succ_adv_x.append(adv_x)
        succ_ori_x.append(test_x)
    print("[INFO] #%d adversarial example is done!\n" % i)
