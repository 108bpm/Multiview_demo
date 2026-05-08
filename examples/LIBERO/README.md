# 🚀 LIBERO Evaluation

This document provides instructions for reproducing our **experimental results** with LIBERO.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO` environment and dependencies.  
2. Running the evaluation by launching services in both `Multi-view-VLA` and `LIBERO` environments.  

Inference requires at least **40 GB** of VRAM. 

---


## ⬇️ 0. Download Checkpoints

Please download Checkpoint from [modelscope](https://www.modelscope.cn/models/junjxiao/Multi-view-VLA) or [huggingface](https://www.huggingface.co/junjin0/Multi-view-VLA) to `./checkpoints`.


---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `LIBERO` environment.  



Afterwards, inside the `LIBERO` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **Multi-view-VLA environment**: runs the inference server.  
- **LIBERO environment**: runs the simulation.  

### Step 1. Start the server (Multi-view-VLA environment)

In the first terminal, activate the `Multi-view-VLA` conda environment and run:  

```bash
bash examples/LIBERO/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO/eval_files/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO environment)

In the second terminal, activate the `LIBERO` conda environment and run:  

```bash
bash examples/LIBERO/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.


---

# 🚀 LIBERO Training

## 📦 Step 0: Download the training dataset
Download the datasets to `./benchmark/libero`:
- [LIBERO-spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

And move `modality.json` to each `$LEROBOT_LIBERO_DATA/subset/meta/modality.json`.

You can also download our cached multi-view features of LIBERO from [modelscope](https://www.modelscope.cn/datasets/junjxiao/libero_mv_feats) or [huggingface](https://www.huggingface.co/datasets/junjin0/libero_mv_feats), then specify `framework.image_edit_model.read_from_local` in [./train_files/starvla_cotrain_libero.yaml](./train_files/starvla_cotrain_libero.yaml) as `true`, and set the `datasets.vla_data.mv_data_root_dir` to your path.



## 🚀 Step1: Start Training

Most of the required training files have been organized in [train_files](train_files).  

Please run the following command to start training, the total batch size is `8x8`:

```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```
⚠️ **Note:** Please ensure that you specify the correct path in `examples/LIBERO/train_files/run_libero_train.sh`

