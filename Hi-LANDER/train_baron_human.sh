echo "Start Time : "
date -R

python train_multi.py --train_data baron_mouse_bar,biase_bar,darmanis_bar,deng_bar,romanov_bar,goolam_bar,zeisel_bar,klein_bar,li_bar,pbmc_68k_bar,segerstolpe_bar,shekhar_mouse_retina_bar,tasic_bar,xin_bar \
  --model_filename  checkpoint/baron_human_bar.ckpt --knn_k 10,5,3 --levels 2,3,4 --hidden 512 --epochs 1000 \
  --lr 0.01 --batch_size 4096 --num_conv 1 --gat --balance --consider_batch > logs/baron_human_meta.log

echo "End Time : "
date -R
