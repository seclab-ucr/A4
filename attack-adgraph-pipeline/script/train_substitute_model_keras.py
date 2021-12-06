#!/usr/bin/env python
# coding: utf-8

# @Shitong Zhu, SRA 2019
# This script trains the model using supplied dataset and Keras

import time
import tensorflow as tf
from tensorflow import keras

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.utils import class_weight

from imblearn.datasets import make_imbalance

from collections import Counter

import logging

import argparse

BASE_DATA_DIR = "../data/"
BASE_REPORT_DIR = "../report/"
BASE_MODEL_DIR = "../model/"

LABEL_DICT = {'True': 'AD', 'False': 'NONAD'}

parser = argparse.ArgumentParser(
    description='Parse arguments (see description in source file)')
parser.add_argument('--train-file', type=str)
parser.add_argument('--test-file', type=str)
parser.add_argument('--num-epoch', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--balance-dist', type=bool)
parser.add_argument('--input-size', type=int, default=312)
args = parser.parse_args()


def dataframe_to_ndarray(dataframe, test_set_size=None):
    dataframe = shuffle(dataframe)
    labels = dataframe.pop('CLASS')
    features = dataframe
    if test_set_size:
        return features.to_numpy()[:-test_set_size], \
            labels.to_numpy()[:-test_set_size], \
            features.to_numpy()[-test_set_size:], \
            labels.to_numpy()[-test_set_size:]
    else:
        return features.to_numpy(), labels.to_numpy()


def train_fn(model,
             train_x,
             train_y,
             test_x=None,
             test_y=None,
             train_balance=False,
             test_balance=False,
             batch_size=10,
             num_epochs=30,
             verbal=True,
             logger=None):

    _BATCH_SIZE = batch_size
    _NUM_EPOCHS = num_epochs
    _FAST_TRAIN = True
    history = None
    cw_dict = None

    def balance_data(x, y, ratio={0: 50, 1: 50}, shuffle=True):
        assert len(x) == len(y), "[ERROR] Different list length!"

        if shuffle:
            rand_idx = np.arange(len(x))
            np.random.shuffle(rand_idx)
            x, y = x[rand_idx], y[rand_idx]

        x_, y_ = make_imbalance(x,
                                y,
                                sampling_strategy=ratio,
                                random_state=66)

        return x_, y_

    def generate_balanced_batch(x, y, batch_size, verbal=False):
        while True:
            x_, y_ = balance_data(x, y)
            if verbal:
                print("[INFO] Label counts:", str(Counter(y_)))

            yield(x_[:batch_size], y_[:batch_size])

    use_test_set = test_x is not None and test_y is not None

    if test_balance:
        assert use_test_set, "[ERROR] test_balance flag set but no test set provided!"

        max_sample_size = max_sample_num = min(
            Counter(test_y).values())
        test_x, test_y = make_imbalance(test_x,
                                        test_y,
                                        sampling_strategy={
                                            0: max_sample_size, 1: max_sample_size},
                                        random_state=66)

    if train_balance:
        if _FAST_TRAIN:
            class_weights = class_weight.compute_class_weight(
                'balanced', np.unique(train_y), train_y)
            cw_dict = dict(zip(list(set(train_y)), class_weights))
        else:
            from math import ceil
            history = model.fit_generator(
                generator=generate_balanced_batch(
                    train_x, train_y, _BATCH_SIZE),
                epochs=_NUM_EPOCHS,
                steps_per_epoch=ceil(len(train_x)/_BATCH_SIZE),
                workers=18,
                use_multiprocessing=True,
                verbose=1
            )

    if verbal and _FAST_TRAIN:
        if logger:
            logger.debug("Class weights:" + str(cw_dict))
        print("[INFO] Class weights:", str(cw_dict))

    if _FAST_TRAIN:
        if use_test_set:
            history = model.fit(
                x=train_x,
                y=train_y,
                validation_data=(test_x, test_y),
                batch_size=_BATCH_SIZE,
                epochs=_NUM_EPOCHS,
                class_weight=cw_dict
            )
        else:
            history = model.fit(
                x=train_x,
                y=train_y,
                batch_size=_BATCH_SIZE,
                epochs=_NUM_EPOCHS,
                class_weight=cw_dict
            )

    return history


def evaluate(model,
             test_x,
             test_y,
             batch_size=100,
             balance=False):

    from sklearn.metrics import classification_report

    if balance:
        max_sample_num = min(Counter(test_y).values())
        test_x, test_y = make_imbalance(test_x,
                                        test_y,
                                        sampling_strategy={
                                            0: max_sample_num, 1: max_sample_num},
                                        random_state=66)

    pred_y = model.predict(test_x, batch_size=batch_size, verbose=1)
    pred_y_bool = np.argmax(pred_y, axis=1)

    print(classification_report(test_y, pred_y_bool))
    return classification_report(test_y, pred_y_bool)


def cross_validate(init_model_def,
                   data_x,
                   data_y,
                   shuffle=True,
                   stratified=True,
                   k=10):

    def get_compiled_new_model(model_def):
        # deep copy
        model_clone = keras.models.clone_model(model_def)
        # we need to compile it because clone_model() only
        # copies the architecture of the model
        model_clone.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model_clone

    def train_inner_loop(model, data_x, data_y):
        train_x, test_x = data_x[train_index], data_x[test_index]
        train_y, test_y = data_y[train_index], data_y[test_index]

        # get a new model clone in order to prevent weights over-writting
        # due to pass by reference (in-place model object modifications)
        new_model_clone = get_compiled_new_model(init_model_def)
        # ignore the trained model
        acc = train_fn(
            new_model_clone,
            train_x,
            train_y,
            test_x,
            test_y
        ).history['acc'][-1]
        evaluate(
            new_model_clone,
            test_x,
            test_y
        )
        return acc

    if stratified:
        kf = StratifiedKFold(n_splits=k, shuffle=shuffle)
    else:
        kf = KFold(n_splits=k, shuffle=shuffle)
    kf.get_n_splits(data_x)
    overall_acc = 0.0

    if stratified:
        for train_index, test_index in kf.split(data_x, data_y):
            overall_acc += train_inner_loop(init_model_def, data_x, data_y)
    else:
        for train_index, test_index in kf.split(data_x):
            overall_acc += train_inner_loop(init_model_def, data_x, data_y)

    overall_acc /= k
    print("Cross-validation acc: %f" % (overall_acc))
    return overall_acc


def train(model,
          train_x,
          train_y,
          test_x=None,
          test_y=None,
          restore=False,
          save=False,
          model_path=None,
          train_balance=False,
          num_epochs=30,
          batch_size=10,
          logger=None):

    if (restore != (model_path is not None)) and (save != (model_path is not None)):
        assert False, "[Error] retore/save flag(s) set but model path not provided!"
    if restore and model_path:
        from pathlib import Path
        model_file = Path(model_path)
        if model_file.exists():
            model_is_restored = True
            model.load_weights(model_path)
    else:
        train_acc = train_fn(model,
                             train_x,
                             train_y,
                             test_x=test_x,
                             test_y=test_y,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             train_balance=train_balance,
                             logger=logger)
        if train_acc:
            print("Training accuracy: %f" % train_acc.history['acc'][-1])
    if save and model_path:
        model.save(model_path)


# column_names are automatically inferred from the first row
# dtype is inferred as well
train_dataframe = pd.read_csv(BASE_DATA_DIR + args.train_file)
test_dataframe = pd.read_csv(BASE_DATA_DIR + args.test_file)

data_x, data_y = dataframe_to_ndarray(train_dataframe)
test_x, test_y = dataframe_to_ndarray(test_dataframe)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(args.input_size,)),
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

curr_time = int(time.time())

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUM_EPOCHS = args.num_epoch
BATCH_SIZE = args.batch_size
IS_BALANCED = args.balance_dist

report_fname = BASE_REPORT_DIR + "experiment_training_%d_%s_%s_%s.log" % (
    curr_time, NUM_EPOCHS, BATCH_SIZE, str(IS_BALANCED))
fileHandler = logging.FileHandler(report_fname, 'a')
logger.addHandler(fileHandler)
logger.debug('========== Experimental Result Report %s ==========' %
             str(curr_time))
logger.debug('========== #1 Model Summary ==========')
model.summary(print_fn=logger.debug)
model_fname = BASE_MODEL_DIR + "adgraph_substitude_%d_%s_%s_%s.h5" % (
    curr_time, NUM_EPOCHS, BATCH_SIZE, str(IS_BALANCED))

logger.debug('========== #2 Training Hyperparamters ==========')
logger.info("Number of epochs: %d" % NUM_EPOCHS)
logger.info("Batch size: %d" % BATCH_SIZE)

train(model,
      data_x,
      data_y,
      test_x,
      test_y,
      save=True,
      train_balance=IS_BALANCED,
      num_epochs=NUM_EPOCHS,
      batch_size=BATCH_SIZE,
      model_path=model_fname,
      logger=logger)

logger.info("Model path: %s" % model_fname)

logger.debug('========== #3 Model Evaluation Results ==========')

report = str(evaluate(model,
                      test_x,
                      test_y,
                      batch_size=1,
                      balance=False))

logger.debug(report)
