# MVS Evaluation Benchmark

This is a customized Multi-view Stereo (MVS) evaluation benchmark built on BlendedMVS and GTA-SFM.

## Log

 - [2022-11-29] Create initial repository.

## To-do

 - Upload evaluation code of BlendedMVS.
 - Upload our processed MVS dataset for BlendedMVS evaluation.
 - Upload our processed ground truth for BlendedMVS evaluation.
 - Upload evaluation code of GTA-SFM.
 - Upload our processed ground truth for GTA-SFM evaluation.
 - Upload our processed ground truth for GTA-SFM evaluation.
 - Upload evaluation demo of a pretrained CascadeMVSNet.

## Brief Introduction

Following [MVSNet](https://github.com/YoYo000/MVSNet), [DTU](https://github.com/YoYo000/MVSNet) and [Tanks\&Temples](https://www.tanksandtemples.org/) are the most popular evaluation benchmarks in recent researches. In this repository, we provide some additional evaluation benchmarks for measuring the performance of MVS algorithm in 3D reconstruction. These benchmarks are built based on the evaluation protocal of Tanks\&Temples: F-score, Precision, Recall. The evaluation of DTU and Tanks\&Temples might take half a day, which seems to be tedious for researches. Hence, we implement this repository based on [PyCUDA](https://github.com/inducer/pycuda) which can **accelerate the evaluation process with GPU parallel computation**.
This repository was originally used for supplementary evaluation of our paper and we hope that it can help future researches about MVS 3D reconstruction.

