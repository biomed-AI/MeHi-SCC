![MeHi-SCC](https://user-images.githubusercontent.com/110893478/184567421-4f9dcca8-3ee0-4257-b7f3-14c6d7882bef.png)

# title
================================================================

This folder contains the official code for [Learning Hierarchical Graph Neural Networks for Image Clustering](https://arxiv.org/abs/2107.01319). ##后续要改为本文的链接

## Setup

We use python 3.8. The CUDA version needs to be 11.6. Besides DGL (>=0.5.2), we depend on several packages. To install dependencies using conda:
```bash
conda create -n MeHi-SCC # create env
conda activate MeHi-SCC # activate env
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.6 -c pytorch # install pytorch 1.11 version
conda install -y cudatoolkit=11.6 faiss-gpu=1.7.2 -c pytorch # install faiss gpu version matching cuda 11.6
pip install future==0.18.2 numpy==1.19.5 pandas==1.4.2 tensorflow==2.4.4 umap-learn==0.5.3 scipy==1.5.3 sklearn

pip install dgl==0.8.2 # install dgl

pip install tqdm # install tqdm
git clone https://github.com/yjxiong/clustering-benchmark.git # install clustering-benchmark for evaluation
cd clustering-benchmark
python setup.py install
cd ../
```

# Hilander

这里要添加Hilander的作用

## Data

The datasets used for training and testing are hosted by the following services.

[BaiduPan](https://pan.baidu.com/s/11t4Likcz-Yj0kMbdYSjqOA) (pwd: 6ylh)

You should unpack the downloaded data and put it under 'hi-lander/data'

## Training

We provide training script as an example.

For training on baron_human, one can run

```bash
cd Hi-LANDER
bash train_baron_human.sh
```

The trained model will be saved in 'Hi-LANDER/checkpoint/' as a file with '.ckpt' suffix.


## Testing

We provide testing script as an example.

```bash
cd Hi-LANDER
bash test_multi_baron_human_bar.sh
```
The results will be saved in 'Hi-LANDER/results/' and 'centre/'


# MeHi-SCC

After Hilander, we go back to the MeHi-SCC directory and run the script we provided.

## Data

The datasets used for MeHi-SCC are hosted by the following services.

[BaiduPan](https://pan.baidu.com/s/1EXgsVMNyjegV6wrDdmw0fw) (pwd: dl82)

You should unpack the downloaded data and put it under 'data/'

## Training & Testing

In addition, we provided some pre-training results under 'centre/' to get the best clustering results.

We provide a script as an example.

For training & testing on baron_human, one can run
```bash
bash test_multi_baron_human.sh
```

You can then find the resulting visualization in 'figures/'
