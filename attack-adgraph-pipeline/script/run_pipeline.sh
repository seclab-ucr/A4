#!/bin/sh
split() {
  echo "[INFO] Spliting the raw/original dataset..."
  python3 split_ds.py --ds-fpath $1 \
                      --target-ds-fpath $2 \
                      --training-ds-fpath $3 \
                      --sample-size $4
}

classify() {
  echo "[INFO] Classifying dataset and dumping output labels..."
  if [ -z "$3" ]
  then
    python3 classifier.py --dataset-fpath $1 \
                          --suffix $2
  else
    python3 classifier.py --dataset-fpath $1 \
                          --suffix $2 \
                          --dump-mode
  fi
}

append() {
  echo "[INFO] Appending labels from Weka/Sklearn model output..."
  python3 append_confidence_to_training_data.py --original-ds $1 \
                                                --model-output $2 \
                                                --appended-label-file $3
}

preprocess() {
  echo "[INFO] Preprocessing training dataset..."
  if [ -z "$4" ]
  then
    python3 preprocess_dataset.py --rescale-feature-names $1 \
                                  --precomputed-stats $2 \
                                  --original-ds $3 \
                                  --dump-untrimmed-label \
                                  --dump-untrimmed-no-label \
                                  --dump-trimmed-label \
                                  --dump-trimmed-no-label
  else
    python3 preprocess_dataset.py --rescale-feature-names $1 \
                                  --precomputed-stats $2 \
                                  --original-ds $3 \
                                  --suffix $4 \
                                  --dump-untrimmed-label \
                                  --dump-untrimmed-no-label \
                                  --dump-trimmed-label \
                                  --dump-trimmed-no-label
  fi
}

shuffle() {
  echo "[INFO] Shuffling and partitioning dataset"
  python3 shuffle_partition_datasets.py --test-set-size $1 \
                                        --trimmed-w-label $2 \
                                        --trimmed-wo-label $3 \
                                        --untrimmed-w-label $4 \
                                        --untrimmed-wo-label $5
}

train() {
  echo "[INFO] Training substitude model"
  python3 train_substitute_model_keras.py --train-file $1 \
                                          --test-file $2 \
                                          --num-epoch $3 \
                                          --batch-size $4
}

attack() {
  echo "[INFO] Generating adversarial examples"
  python3 pgd_adgraph_foolbox_verifiable.py --target-file $1 \
                                            --unnorm-target-file $2 \
                                            --feature-defs $3 \
                                            --feature-idx $4 \
                                            --unnorm-feature-idx $5\
                                            --target-gt-file $6 \
                                            --model-file $7 \
                                            --preprocess-feature-defs $8 \
					    --start-id $9 \
					    --augment-pgd \
					    --end-id $10 \
					    --feature-set $11 \
					    --browser-id $12
}

generate_def() {
  echo "[INFO] Generating feature definitions"
  python3 generate_all_def.py --unnormalized-dataset $1 \
                              --base-feature-def $2
}

if [ $1 = "end2end" ]
then
  echo "[INFO] Running the entire pipeline..."

  echo "[INFO] Step #00"
  generate_def dataset_1203.csv \
               base_feature_info.csv

  echo "[INFO] Step #0a"
  split dataset_1203.csv \
        target_dataset.csv \
        training_dataset.csv \
        10000

  echo "[INFO] Step #0bi"
  classify training_dataset.csv \
           training \
           dump_mode

  echo "[INFO] Step #0bii"
  classify training_dataset.csv \
           training

  echo "[INFO] Step #1"
  append training_dataset.csv \
         training_model_output.txt \
         training_dataset_gt.csv

  echo "[INFO] Step #2"
  preprocess column_names_to_rescale.txt \
             col_stats_for_unnormalization.csv \
             training_dataset.csv

  echo "[INFO] Step #3"
  shuffle 50000 \
          hand_preprocessed_trimmed_label.csv \
          hand_preprocessed_trimmed_no_label.csv \
          hand_preprocessed_untrimmed_label.csv \
          hand_preprocessed_untrimmed_no_label.csv

  echo "[INFO] Step #4"
  train hand_preprocessed_trimmed_label_augmented_train_set.csv \
        hand_preprocessed_trimmed_label_augmented_test_set.csv \
        30 \
        100

  echo "[INFO] Step #5"
  preprocess column_names_to_rescale.txt \
             col_stats_for_unnormalization.csv \
             target_dataset.csv \
             _target

  echo "[INFO] Step #6"
  attack hand_preprocessed_untrimmed_no_label_target.csv \
         target_dataset.csv \
         preprocessed_adgraph_alexa_10k_feature_defs.txt \
         trimmed_wo_class_feature_idx.csv \
         unnormalized_feature_idx.csv \
         hand_preprocessed_untrimmed_label_target.csv \
         adgraph_substitude_1614499579_30_100_None.h5 \
         hand_preprocessing_defs.csv \
  	 0

elif [ $1 = "generate_def" ]
then
  generate_def dataset_1111.csv \
               base_feature_info.csv
elif [ $1 = "split" ] 
then
  split dataset_1111.csv \
        target_dataset.csv \
        training_dataset.csv \
        10000
elif [ $1 = "classify" ]
then
  classify training_dataset.csv \
           training
elif [ $1 = "shuffle" ]
then
  shuffle 50000 \
          hand_preprocessed_trimmed_label.csv \
          hand_preprocessed_trimmed_no_label.csv \
          hand_preprocessed_untrimmed_label.csv \
          hand_preprocessed_untrimmed_no_label.csv
elif [ $1 = "append" ]
then
  append training_dataset.csv \
         training_model_output_target.txt \
         training_dataset_gt.csv
elif [ $1 = "preprocess-dataset" ]
then
  preprocess column_names_to_rescale.txt \
             col_stats_for_unnormalization.csv \
             training_dataset.csv
elif [ $1 = "preprocess-target" ]
then
  preprocess column_names_to_rescale.txt \
             col_stats_for_unnormalization.csv \
             target_dataset.csv \
             _target
elif [ $1 = "train" ]
then
  train hand_preprocessed_trimmed_label_augmented_train_set.csv \
        hand_preprocessed_trimmed_label_augmented_test_set.csv \
        10 \
        128
elif [ $1 = "attack" ]
then
  attack hand_preprocessed_untrimmed_no_label_target.csv \
         target_dataset.csv \
         preprocessed_adgraph_alexa_10k_feature_defs.txt \
         trimmed_wo_class_feature_idx.csv \
         unnormalized_feature_idx.csv \
         hand_preprocessed_untrimmed_label_target.csv \
         adgraph_substitude_1614499579_30_100_None.h5 \
         hand_preprocessing_defs.csv \
	 $2 \
	 $3 \
	 $4 \
	 $5
elif [ $1 = "clean" ]
then
  rm ../data/hand_preprocessed_*
  rm ../data/*_dataset*
  rm ../data/*.txt
  rm ../model/*
  rm ../report/*
fi
