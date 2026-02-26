# MLPerf Whisper Inference - Complete Installation Guide                                                                                                                                     
                 
## Table of Contents

- [Supported Configurations](#supported-configurations)
- [Step 1: Clone MLPerf Inference Repository](#step-1-clone-mlperf-inference-repository)
- [Step 2: Install Miniconda](#step-2-install-miniconda)
- [Step 3: Create Conda Environment](#step-3-create-conda-environment)
- [Step 4: Install PyTorch](#step-4-install-pytorch)
- [Step 5: Install vLLM and Dependencies](#step-5-install-vllm-and-dependencies)
- [Step 6: Install MLPerf LoadGen](#step-6-install-mlperf-loadgen)
- [Step 7: Download Whisper Model](#step-7-download-whisper-model)
- [Step 8: Download Dataset](#step-8-download-dataset)
- [Step 9: Configure Environment Variables](#step-9-configure-environment-variables)
  - [L40S Configuration](#l40s-configuration)
  - [H200 Configuration](#h200-configuration)
- [Step 10: Run Benchmark](#step-10-run-benchmark)
- [Expected Performance Results](#expected-performance-results)
- [Verify Results](#verify-results)

---

## Supported Configurations

| | NVIDIA L40S | NVIDIA H200 |
|---|---|---|
| **GPUs** | 2x | 8x |
| **GPU Memory** | 48 GB GDDR6 | 141 GB HBM3e |
| **OS** | RHEL 9.6 | RHEL 9.6 |
| **CUDA** | 13.0 | 12.9 |
| **Python** | 3.12.12 | 3.12.12 |
| **NUMA Nodes** | 2 | 2 |
| **CPU Cores** | 48 (24 per NUMA) | 160 (80 per NUMA) |
| **GPU Interconnect** | PCIe | NV18 (18 NVLinks) |

---

## Step 1: Clone MLPerf Inference Repository

```bash
cd ~
git clone --recurse-submodules https://github.com/mlcommons/inference.git
cd inference
git checkout master
```

---

## Step 2: Install Miniconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

---

## Step 3: Create Conda Environment

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n whisper-gpu python=3.12
conda activate whisper-gpu
```

**Verify:**
```bash
python --version
# Expected: Python 3.12.x
```

---

## Step 4: Install PyTorch

```bash
pip install --no-cache-dir \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0
```

**Verify:**
```bash
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
```

---

## Step 5: Install vLLM and Dependencies

```bash
pip install --no-cache-dir --index-url https://pypi.org/simple vllm==0.15.1

pip install --no-cache-dir \
    pandas==2.2.2 \
    librosa==0.10.2 \
    numpy==2.0.1

pip install --no-cache-dir openai-whisper==20250625

pip install --no-cache-dir \
    toml==0.10.2 \
    unidecode==1.3.8 \
    inflect==7.3.1 \
    setuptools-scm \
    py-libnuma==1.2
```

**Verify:**
```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

---

## Step 6: Install MLPerf LoadGen

```bash
cd ~/inference/loadgen
pip install --no-cache-dir --index-url https://pypi.org/simple -e .
python -c "import mlperf_loadgen as lg; print('LoadGen installed successfully')"
```

---

## Step 7: Download Whisper Model

```bash
cd ~/inference/speech2text

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d ./model/whisper-large-v3 \
  https://inference.mlcommons-storage.org/metadata/whisper-model.uri
```

**Verify:**
```bash
ls -lh model/whisper-large-v3/
```

---

## Step 8: Download Dataset

```bash
cd ~/inference/speech2text

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d ./data \
  https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri
```

**Note:** Check the actual manifest filename:
```bash
ls -lh data/data/*.json
```

---

## Step 9: Configure Environment Variables

### Common Settings (both GPUs)

```bash
cd ~/inference/speech2text

export WORKSPACE_DIR=$(pwd)
export DATA_DIR=${WORKSPACE_DIR}/data
export MODEL_PATH=${WORKSPACE_DIR}/model/whisper-large-v3
export MANIFEST_FILE="${DATA_DIR}/dev-all-repack.json"
export SCENARIO="Offline"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### L40S Configuration

**1-GPU:**
```bash
export CUDA_VISIBLE_DEVICES=0
export NUM_INSTS=1
export NUM_NUMA_NODES=1
export INSTS_PER_NODE=1
export CORES_PER_INST=24
export START_CORES="0"
```

**2-GPU (recommended):**
```bash
export CUDA_VISIBLE_DEVICES=0,1
export NUM_INSTS=2
export NUM_NUMA_NODES=2
export INSTS_PER_NODE=1
export CORES_PER_INST=24
export START_CORES="0,24"
```

### H200 Configuration

**1-GPU (recommended):**
```bash
export NUM_CORES=160
export NUM_NUMA_NODES=1
export INSTS_PER_NODE=1
export NUM_INSTS=1
export CORES_PER_INST=40
export OMP_NUM_THREADS=40
export START_CORES="0"
```

**8-GPU (recommended):**
```bash
export NUM_CORES=160
export NUM_NUMA_NODES=2
export INSTS_PER_NODE=4
export NUM_INSTS=8
export CORES_PER_INST=20
export OMP_NUM_THREADS=20
export START_CORES="0,80"
```

<details>
<summary><b>H200 GPU Topology Reference</b></summary>

```
GPU0-3: NUMA node0 (CPUs 0-79),  NV18 interconnect
GPU4-7: NUMA node1 (CPUs 80-159), NV18 interconnect
```

| Config | GPUs | NUMA | Cores/inst | START_CORES |
|---|---|---|---|---|
| Single | GPU0 | 1 | 40 | `"0"` |
| Dual (same NUMA) | GPU0,1 | 1 | 20 | `"0,20"` |
| Dual (cross NUMA) | GPU0,4 | 2 | 40 | `"0,80"` |
| 8-GPU | All | 2 | 10 | `"0,10,20,30,80,90,100,110"` |

</details>

### Verify Dataset and Model

```bash
python -c "import json; data=json.load(open('$MANIFEST_FILE')); print(f'Dataset has {len(data)} samples')"
# Expected: Dataset has 1633 samples
```

---

## Step 10: Run Benchmark

### Apply the patch 
#### For L40S
```bash
git apply l40s.patch
```
#### For H200
```bash
git apply h200.patch
```

```bash
mkdir -p logs
```

### Performance Run

```bash
python reference_mlperf.py \
    --dataset_dir ${DATA_DIR} \
    --model_path ${MODEL_PATH} \
    --manifest ${MANIFEST_FILE} \
    --scenario ${SCENARIO} \
    --log_dir logs/performance \
    --num_workers ${NUM_INSTS}
```

### Accuracy Run

```bash
python reference_mlperf.py \
    --dataset_dir ${DATA_DIR} \
    --model_path ${MODEL_PATH} \
    --manifest ${MANIFEST_FILE} \
    --scenario ${SCENARIO} \
    --log_dir logs/accuracy \
    --num_workers ${NUM_INSTS} \
    --accuracy
```

### Monitor (separate terminal)

```bash
nvidia-smi -l
htop
```

---

## Expected Performance Results

| Metric | L40S (2 GPU) | H200 (8 GPU) |
|---|---|---|
| Samples/sec | ~48 | ~485 |
| Tokens/sec | ~3,637 | ~36,378 |
| GPU Utilization | 90-99% | 90-99% |
| Duration | ~10-12 min | ~35 min |
| Result | VALID | VALID |

### Common Configuration

- Dataset: 1,633 samples (~10.91 hours of audio)
- Precision: bfloat16
- Max Model Length: 448 tokens
- GPU Memory Utilization: 95%
- Max Batched Tokens: 32,000
- Max Sequences: 256
- Additionally for L40S we set   kv_cache_dtype="fp8_e4m3",

---

## Verify Results

```bash
# Performance summary
cat logs/performance/mlperf_log_summary.txt
grep "Result is" logs/performance/mlperf_log_summary.txt

# Accuracy (if accuracy run was performed)
python accuracy_eval.py --log_dir logs/accuracy --dataset_dir . --manifest data/dev-all-repack.json
cat logs/accuracy/accuracy.txt
```

---

**Last Updated**: 2026-02-25
