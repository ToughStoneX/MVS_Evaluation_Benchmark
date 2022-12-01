# MVS Evaluation Benchmark

This is a customized Multi-view Stereo (MVS) evaluation benchmark built on BlendedMVS and GTA-SFM.

## Log

 - [2022-11-29] Create initial repository.
 - [2022-12-01] Upload evaluation demo of a pretrained CascadeMVSNet.
 - [2022-12-01] Upload evaluation code of BlendedMVS.
 - [2022-12-01] Upload our processed MVS dataset for BlendedMVS evaluation.
 - [2022-12-01] Upload our processed ground truth for BlendedMVS evaluation.
 - [2022-12-01] Upload evaluation code of GTA-SFM.
 - [2022-12-01] Upload our processed ground truth for GTA-SFM evaluation.
 - [2022-12-01] Upload our processed ground truth for GTA-SFM evaluation.

## To-do

 - Run several MVS models on these benchmarks and create a leaderboard comparison.
 

## Brief Introduction

Following [MVSNet](https://github.com/YoYo000/MVSNet), [DTU](https://github.com/YoYo000/MVSNet) and [Tanks\&Temples](https://www.tanksandtemples.org/) are the most popular evaluation benchmarks in recent researches. In this repository, we provide some additional evaluation benchmarks for measuring the performance of MVS algorithm in 3D reconstruction. These benchmarks are built based on the evaluation protocal of Tanks\&Temples: F-score, Precision, Recall. The evaluation of DTU and Tanks\&Temples might take half a day, which seems to be tedious for researches. Hence, we implement this repository based on Pytorch with CUDA support which can **accelerate the evaluation process with GPU parallel computation**.
This repository was originally used for supplementary evaluation of our paper and we hope that it can help future researches about MVS 3D reconstruction.

## Data

 - You can download the BlendedMVS MVS dataset from [here](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/datasets/blendedmvs/BlendedMVS.zip).
 - Unzip the compressed file with `unzip BlendedMVS.zip`.
 - You can download our processed ground truth point clouds of BlendedMVS from [here](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/share/mvs_evaluation_benchmark/blendedmvs_ground_truth/Points.tar.gz).
 - Unzip the compressed file with `tar -zxvf Points.tar.gz`.
 - You download our processed GTASFM MVS dataset from [here](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/datasets/gta_sfm/gta_sfm_clean.tar.gz).
 - Unzip the compressed file with `tar -zxvf gta_sfm_clean.tar.gz`.
 - You can download our processed ground truth point clouds of GTASFM from [here](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/share/mvs_evaluation_benchmark/gtasfm_ground_truth/Points.zip).
 - Unzip the compressed file with `tar -zxvf Points.zip`.

## Environment

To run the evaluation code, you may require:
> Pytorch 1.8 or higher \
> importlib 

To run the demo of the whole pipeline to reconstruct the model with MVS model first and evaluate the performance, you may require:
> Pytorch 1.8 or higher \
> importlib \
> plyfile \
> numpy \
> torchvision \
> pillow

The [fusibile](https://github.com/kysucix/fusibile) tool used for 3D reconstruction needs to be builded to fit the computation architecture of GPU.
Uncomment the code in `fusibile/CMakeLists.txt` according to what GPU you are using.
```
# 1080 Ti
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=sm_60)

# 2080 Ti
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75)

# 3090
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=sm_86)

# V100-32G
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=sm_70)
```
And build the fusibile by:
```
cd fusibile
mkdir build
cd build
cmake ..
make
```

## How to use?

The template evaluation scripts of BlendedMVS and GTASFM are respectively provided in `scripts/demo_blendedmvs_eval.sh` and `scripts/demo_gtasfm_eval.sh`.

Taking `scripts/demo_gtasfm_eval.sh` as an example, you can set `PCS_PATH` as the path of reconstructed point clouds of all test scenes, and set `GTS_PATH` as the path of downloaded ground truth point clouds. 
In analogy, you can modify the `scripts/demo_blendedmvs_eval.sh` in the same way.
```
PCS_PATH="outputs/gtasfm/eval/"
# [NOTE] Modify "GTS_PATH" to the path of ground truth files saved in your machine
GTS_PATH="/home/xhb/datasets/evaluation_gtasfm/Points/stl"

CUDA_VISIBLE_DEVICES=0 python eval_blendedmvs.py \
    --prefix "mvsnet" \
    --pcs_path ${PCS_PATH} \
    --gts_path ${GTS_PATH}
```

## Demo

We provide a template demo for reconstructing 3D point clouds with MVS network and conducting evaluation on our benchmarks.

### BlendedMVS

Modify the `BLENDEDMVS_PATH` to the path of downloaded BlendedMVS dataset in your machine. And run the reconstruction script:
```
bash scripts/demo_blendedmvs_recon.sh
```

Then, run the evaluation scripts, and the results will be saved in `results_blendedmvs.txt` and `results_blendedmvs.json`.
```
bash scripts/demo_blendedmvs_eval.sh
```

### GTASFM

Modify the `GTASFM_PATH` to the path of downloaded GTASFM dataset in your machine. And run the reconstruction script:
```
bash scripts/demo_gtasfm_recon.sh
```

Then, run the evaluation scripts, and the results will be saved in `results_gtasfm.txt` and `results_gtasfm.json`.
```
bash scripts/demo_gtasfm_recon.sh
```


## Format of Results

### BlendedMVS

Results of evaluation on BlendedMVS saved in `results_blendedmvs.txt`. The mean values of `F-score`, `Precision`, and `Recall` is used as overall measure of reconstruction quality.
> [number of scenes] \
> [scene name] [F-score] [Precision] [Recall] \
> ...... \
> mean [F-score] [Precision] [Recall]

Example:
> 7 \
> 5b7a3890fc8fcf6781e2593a 0.2186 0.1726 0.2979 \
> 5c189f2326173c3a09ed7ef3 0.1651 0.1125 0.3102 \
> 5b950c71608de421b1e7318f 0.4386 0.3699 0.5385 \
> 5a6400933d809f1d8200af15 0.5490 0.5027 0.6047 \
> 59d2657f82ca7774b1ec081d 0.3340 0.3275 0.3407 \
> 5ba19a8a360c7c30c1c169df 0.5094 0.9073 0.3541 \
> 59817e4a1bd4b175e7038d19 0.4311 0.8461 0.2892 \
> mean 0.3779 0.4627 0.3907

### GTASFM

Results of evaluation on BlendedMVS saved in `results_gtasfm.txt`. The mean values of `F-score`, `Precision`, and `Recall` is used as overall measure of reconstruction quality.
> [number of scenes] \
> [scene name] [F-score] [Precision] [Recall] \
> ...... \
> mean [F-score] [Precision] [Recall]

Example:
> 18 \
> 20190127_154841 0.3909 0.4918 0.3244 \
> 20190127_160317 0.4356 0.3945 0.4863 \
> 20190127_160737 0.4320 0.5644 0.3499 \
> 20190127_161129 0.4029 0.4709 0.3520 \
> 20190127_161419 0.3003 0.2232 0.4588 \
> 20190127_161922 0.5562 0.6045 0.5150 \
> 20190127_162256 0.3947 0.3751 0.4164 \
> 20190127_162742 0.3372 0.2978 0.3885 \
> 20190127_163215 0.4500 0.4278 0.4746 \
> 20190127_163742 0.2646 0.4350 0.1901 \
> 20190127_164516 0.4432 0.4678 0.4211 \
> 20190127_164840 0.3960 0.4050 0.3875 \
> 20190127_165303 0.4906 0.6512 0.3936 \
> 20190127_165928 0.3216 0.3447 0.3014 \
> 20190127_171304 0.4185 0.5097 0.3549 \
> 20190221_174234 0.4645 0.4943 0.4380 \
> 20190221_174830 0.4435 0.4741 0.4166 \
> 20190221_175345 0.5023 0.5376 0.4713 \
> mean 0.4136 0.4538 0.3967


## Citaion

Our paper related to this repository is still in submission.
If you find this code useful, please cite the related works.
```
@article{xu2022semi,
  title={Semi-supervised Deep Multi-view Stereo},
  author={Xu, Hongbin and Zhou, Zhipeng and Cheng, Weitao and Sun, Baigui and Li, Hao and Kang, Wenxiong},
  journal={arXiv preprint arXiv:2207.11699},
  year={2022}
}
```