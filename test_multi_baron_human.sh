#!/bin/bash
for dataset in "baron_human"
do
  for k in 3 4 5 6 7 8 9 10
  do
    for l in 2 3 4 5
    do
      for t in 0.1
      do
        for ce in 0.01
        do
          echo "Dataset: $dataset"
          echo "KNN_K : $k"
          echo "LEVELS : $l"
          echo "TAU : $t"
          echo "ce_loss : $ce"
          python MeHi_SCC.py --name $dataset --model_filename Hi-LANDER/checkpoint/baron_human_bar.ckpt \
          --knn_k $k --level $l --tau $t --ce_loss $ce --gat --early_stop --GCNII --cuda_no 2
        done
      done
    done
  done
done
echo "Done!"
