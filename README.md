# MVS Evaluation Benchmark

This is a customized Multi-view Stereo (MVS) evaluation benchmark built on BlendedMVS and GTA-SFM.

## Log

 - [2022-11-29] Create initial repository.
 - [2022-12-01] Upload evaluation demo of a pretrained CascadeMVSNet.
 - [2022-12-01] Upload evaluation code of BlendedMVS.
 - [2022-12-01] Upload our processed MVS dataset for BlendedMVS evaluation.
 - [2022-12-01] Upload our processed ground truth for BlendedMVS evaluation.

## To-do

 - Upload evaluation code of GTA-SFM.
 - Upload our processed ground truth for GTA-SFM evaluation.
 - Upload our processed ground truth for GTA-SFM evaluation.
 

## Brief Introduction

Following [MVSNet](https://github.com/YoYo000/MVSNet), [DTU](https://github.com/YoYo000/MVSNet) and [Tanks\&Temples](https://www.tanksandtemples.org/) are the most popular evaluation benchmarks in recent researches. In this repository, we provide some additional evaluation benchmarks for measuring the performance of MVS algorithm in 3D reconstruction. These benchmarks are built based on the evaluation protocal of Tanks\&Temples: F-score, Precision, Recall. The evaluation of DTU and Tanks\&Temples might take half a day, which seems to be tedious for researches. Hence, we implement this repository based on Pytorch with CUDA support which can **accelerate the evaluation process with GPU parallel computation**.
This repository was originally used for supplementary evaluation of our paper and we hope that it can help future researches about MVS 3D reconstruction.

## Data

 - You can download the BlendedMVS MVS dataset from [here]().
 - You can download our processed ground truth point clouds of BlendedMVS from [here](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/share/mvs_evaluation_benchmark/blendedmvs_ground_truth/Points.tar.gz).
 - You download our processed GTASFM MVS dataset from [here]().
 - You can download our processed ground truth point clouds of GTASFM from [here](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/share/mvs_evaluation_benchmark/gtasfm_ground_truth/Points.zip).

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
