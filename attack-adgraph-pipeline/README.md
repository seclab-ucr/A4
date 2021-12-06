# Generating Adversarial Examples Under Constraints

To run the pipeline:
```
python3 pgd_adgraph_foolbox_verifiable.py --target-file hand_preprocessed_untrimmed_no_label_target.csv --target-gt-file hand_preprocessed_untrimmed_label_target_gt.csv --unnorm-target-file target_dataset.csv --model-file adgraph_substitude_1570169176_30_100_None.h5 --feature-defs preprocessed_adgraph_alexa_10k_feature_defs.txt --feature-idx trimmed_wo_class_feature_idx.csv --preprocess-feature-defs hand_preprocessing_defs_new.csv --unnorm-feature-idx unnormalized_feature_idx.csv --target-set-size 10000 --start-idx 0 --end-idx 300 --augment-pgd
```

## 0x00 Intro
This repo holds code for generating adversarial examples against AdGraph [1]
## 0x01 Preprocessing Dataset
First off, let's preprocess the dataset so they fit our next training and attack phases
### For training
We use neural network to train the local surrogate model. To better itimate the decision boundary of remote random forest model, we need to preprocess the dataset. Our preprocessing consists of two parts: one-hot encoding and value normalization.
### For carrying out attack
Since our attack is also performed based on gradients derived from the NN model, it's necessary to make sure the data points provided to PGD are also preprocessed.

[1] AdGraph: A Graph-Based Approach to Ad and Tracker Blocking
Umar Iqbal, Peter Snyder, Shitong Zhu, Benjamin Livshits, Zhiyun Qian, and Zubair Shafiq
IEEE Symposium on Security & Privacy (S&P), 2020
