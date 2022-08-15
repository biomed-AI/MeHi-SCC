#!/bin/bash
for dataset in "baron_human_bar"
do
  for k in 3 4 5 6 7 8 9 10
  do
    for l in 1 2 3 4 5
    do
      for t in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
      do
        echo "Dataset: $dataset"
        echo "KNN_K : $k"
        echo "LEVELS : $l"
        echo "TAU : $t"
        python test_multi.py --test_data $dataset --affl Hi-LANDER_meta \
        --knn_k $k --tau $t --level $l --threshold prob --hidden 512 --num_conv 1 --gat --batch_size 4096 \
        --early_stop --consider_batch --cuda_no 0
      done
    done
  done
done
echo "Done!"