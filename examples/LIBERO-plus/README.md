# 🚀 LIBERO-plus zero shot Evaluation

This document provides instructions for reproducing our **experimental results** with LIBERO-plus.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO-plus` environment and dependencies.  
2. Running the evaluation by launching services in both `Multi-view-VLA` and `LIBERO-plus` environments.  

Inference requires at least **40 GB** of VRAM.

---


## ⬇️ 0. Download Checkpoints

Please download Checkpoint from [modelscope](https://www.modelscope.cn/models/junjxiao/Multi-view-VLA) or [huggingface](https://www.huggingface.co/junjin0/Multi-view-VLA) to `./checkpoints`.

---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO-plus repository](https://github.com/sylvestf/LIBERO-plus) to install the base `LIBERO-plus` environment.  



Afterwards, inside the `LIBERO-plus` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **Multi-view-VLA environment**: runs the inference server.  
- **LIBERO-plus environment**: runs the simulation.  

### Step 1. Start the server (Multi-view-VLA environment)

In the first terminal, activate the `Multi-view-VLA` conda environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/run_policy_server.sh
```

⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `examples/LIBERO-plus/eval_files/run_policy_server.sh`  


---

### Step 2. Start the simulation (LIBERO-plus environment)

In the second terminal, activate the `LIBERO-plus` conda environment and run:  

```bash
bash examples/LIBERO-plus/eval_files/eval_libero.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `eval_libero.sh` to load action unnormalization stats. 

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.


---

⚠️ **Note:** Since LIBERO-plus has 10,030 tasks, completing all the evaluations will take an extremely long time. It is recommended to run multiple model instances in parallel for the evaluations. We provide code and scripts for parallel testing on cluster `./parallel_eval/run_nebula_libero_plus`. Please modify them to fit your own cluster.