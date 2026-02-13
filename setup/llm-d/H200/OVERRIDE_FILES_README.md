# GPT-OSS-120B Deployment Override Files for LLM-D v0.5.0

This directory contains Helm values override files for deploying GPT-OSS-120B with llm-d v0.5.0 in two configurations: **Server** and **Offline**.

## Files

1. **`deploy_gptoss120b_v050.sh`** - Automated deployment script (recommended)
2. **`override_ms_gptoss120b_model.yaml`** - Model artifacts configuration (common to both server and offline)
3. **`override_ms_gptoss120b_server_v050.yaml`** - Server mode configuration (for MLPerf server scenario)
4. **`override_ms_gptoss120b_offline_v050.yaml`** - Offline mode configuration (for MLPerf offline scenario)

## Quick Start

### Option 1: Automated Deployment Script (Recommended)

The easiest way to deploy is using the provided deployment script:

```bash
# Deploy in server mode
./deploy_gptoss120b_v050.sh server

# Deploy in offline mode
./deploy_gptoss120b_v050.sh offline

# Dry run (see what would be deployed without actually deploying)
./deploy_gptoss120b_v050.sh server --dry-run
```

The script automatically:
1. Clones the llm-d repository (if not already present)
2. Checks out v0.5.0
3. Copies override files to the correct location
4. Deploys infrastructure, GAIE, and model service in order
5. Shows pod status after deployment

**Environment Variables**:
- `LLMD_DIR` - Where to clone llm-d (default: `/tmp/llm-d`)
- `NAMESPACE` - Kubernetes namespace (default: `llm-d-bench`)
- `RELEASE_NAME_POSTFIX` - Helm release postfix (default: `inference-scheduling`)

Example with custom settings:
```bash
NAMESPACE=my-namespace LLMD_DIR=/home/user/llm-d ./deploy_gptoss120b_v050.sh server
```

### Option 2: Manual Deployment

If you prefer manual control or want to understand each step:

### Prerequisites

- Kubernetes cluster with Istio 1.28.1 GA or later
- At least 128 CPUs available on worker nodes (16 CPUs × 8 replicas)
- 8 NVIDIA GPUs available
- HuggingFace token secret `llm-d-hf-token` created in your namespace
- Model artifacts available at the PVC path specified in `override_ms_gptoss120b_model.yaml`

### Deployment Steps

#### 1. Clone and Setup llm-d Repository

```bash
git clone https://github.com/llm-d/llm-d
cd llm-d
git checkout v0.5.0
cd guides/inference-scheduling/
```

#### 2. Copy Override Files

Copy the three override YAML files to the `guides/inference-scheduling/` directory:

```bash
# Copy all three files here:
# - override_ms_gptoss120b_model.yaml
# - override_ms_gptoss120b_server_v050.yaml (for server mode)
# - override_ms_gptoss120b_offline_v050.yaml (for offline mode)
```

#### 3. Deploy Infrastructure and GAIE (First Time Only)

If this is your first deployment, install the infrastructure components first:

```bash
# Apply infrastructure (Istio gateway)
helmfile -f helmfile.yaml.gotmpl -l name=infra-inference-scheduling apply

# Apply GAIE (Gateway API Inference Extension)
helmfile -f helmfile.yaml.gotmpl -l name=gaie-inference-scheduling apply
```

**Note**: The `-l name=...` selector filters helmfile to operate on only that specific release. The helmfile defines three releases:
- `infra-inference-scheduling` - Infrastructure components
- `gaie-inference-scheduling` - Gateway API Inference Extension
- `ms-inference-scheduling` - Model Service (vLLM)

#### 4. Deploy Model Service

Choose either **Server** or **Offline** configuration:

**For Server Mode (MLPerf Server Scenario):**

```bash
helmfile -f helmfile.yaml.gotmpl -l name=ms-inference-scheduling apply \
  --args '--values override_ms_gptoss120b_server_v050.yaml --values override_ms_gptoss120b_model.yaml --set decode.replicas=8 --set decode.parallelism.tensor=1'
```

**For Offline Mode (MLPerf Offline Scenario):**

```bash
helmfile -f helmfile.yaml.gotmpl -l name=ms-inference-scheduling apply \
  --args '--values override_ms_gptoss120b_offline_v050.yaml --values override_ms_gptoss120b_model.yaml --set decode.replicas=8 --set decode.parallelism.tensor=1'
```

The `-l name=ms-inference-scheduling` selector ensures the override values are **only applied to the model service release**, not to infra or gaie.

## Configuration Details

### Common Configuration (Both Server and Offline)

All configurations share these settings:

- **vLLM Version**: v0.14.1 (llm-d-cuda:v0.5.0 container)
- **Model**: openai/gpt-oss-120b
- **Replicas**: 8 decode pods
- **Tensor Parallelism**: 1 (TP=1 per replica, using 1 GPU each)
- **CPU per Pod**: 16 CPUs (reduced from default 32)
- **Memory per Pod**: 100Gi
- **Prefix Caching**: Disabled (`--no-enable-prefix-caching`)
- **Async Scheduling**: Enabled (`--async-scheduling`)
- **Max Sequences**: 448 (`--max-num-seqs`)
- **CUDA Compatibility Fix**: Mounts emptyDir over `/usr/local/cuda/compat`

### Server vs Offline Configuration Differences

The key differences between server and offline modes:

| Setting | Server Mode | Offline Mode | Why Different? |
|---------|-------------|--------------|----------------|
| **max-num-batched-tokens** | 2048 | 4096 | Offline can handle larger batches |
| **max_cudagraph_capture_size** | 2048 | 8192 | Offline benefits from larger CUDA graphs |

**Server Mode (`override_ms_gptoss120b_server_v050.yaml`):**
- Optimized for latency-sensitive, real-time inference
- Smaller batch sizes (2048 tokens) for faster response times
- Smaller CUDA graph capture size (2048) for quicker compilation
- Suitable for: MLPerf server scenario, production serving workloads

**Offline Mode (`override_ms_gptoss120b_offline_v050.yaml`):**
- Optimized for throughput-oriented batch processing
- Larger batch sizes (4096 tokens) for maximum throughput
- Larger CUDA graph capture size (8192) for better kernel fusion
- Suitable for: MLPerf offline scenario, batch inference jobs

## Model Configuration

**File**: `override_ms_gptoss120b_model.yaml`

```yaml
modelArtifacts:
  uri: "pvc://models-storage/models/openai-gpt-oss-120b"
  name: "openai/gpt-oss-120b"
  size: 500Gi
```

**Note**: Modify the `uri` field to match your actual model storage location:
- For PVC: `pvc://your-pvc-name/path/to/model`
- For HuggingFace: `hf://org/model-name`
- For S3: `s3://bucket-name/path/to/model`

## Technical Details

### Why CPU Reduced to 16?

Default v0.5.0 values specify 32 CPUs per pod. With 8 replicas:
- Default: 32 CPUs × 8 = 256 CPUs required
- Typical nodes: ~160 CPUs available
- **Solution**: Reduce to 16 CPUs × 8 = 128 CPUs (fits within node capacity)

This allows all 8 replicas to schedule on available nodes while still providing adequate CPU resources.

### Why the CUDA Compatibility Fix?

The llm-d-cuda:v0.5.0 container image includes CUDA compatibility libraries built for NVIDIA driver 575.x. If your nodes run a newer driver (e.g., 580.x), these compatibility libraries can cause conflicts.

**Solution**: Mount an emptyDir volume over `/usr/local/cuda/compat` to hide the incompatible libraries. The system will then use the correct driver-native CUDA libraries.

This is automatically applied by the override files via:
```yaml
volumeMounts:
  - mountPath: /usr/local/cuda/compat
    name: hide-cuda-compat
volumes:
  - emptyDir: {}
    name: hide-cuda-compat
```

### Why Disable Prefix Caching?

Prefix caching is disabled (`--no-enable-prefix-caching`) to test the inference scheduler behavior without prefix affinity. This provides:
- Simpler request routing (no prefix-based affinity)
- More even load distribution across replicas
- Easier performance analysis and debugging

For production workloads with repeated prefixes, you may want to enable prefix caching by removing this flag.

## Verification

After deployment, verify the pods are running:

```bash
# Check all pods
kubectl get pods -n llm-d-bench -l app.kubernetes.io/instance=ms-inference-scheduling

# Check specific decode pods
kubectl get pods -n llm-d-bench -l app.kubernetes.io/instance=ms-inference-scheduling,app.kubernetes.io/component=decode

# Verify GPU allocation
kubectl describe pod -n llm-d-bench -l app.kubernetes.io/component=decode | grep -A 5 "nvidia.com/gpu"
```

Expected output: 8 decode pods in Running state, each with 1 GPU allocated.

## Test the Deployment

Test that the inference gateway is responding:

```bash
# Get gateway URL (from within cluster)
GATEWAY_URL="http://infra-inference-scheduling-inference-gateway-istio.llm-d-bench.svc.cluster.local:80"

# Test completions endpoint
curl -X POST "${GATEWAY_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

## Switching Between Server and Offline

To switch between server and offline configurations, simply re-run helmfile with the appropriate override file:

```bash
# Switch to server mode
helmfile -f helmfile.yaml.gotmpl -l name=ms-inference-scheduling apply \
  --args '--values override_ms_gptoss120b_server_v050.yaml --values override_ms_gptoss120b_model.yaml --set decode.replicas=8 --set decode.parallelism.tensor=1'

# Switch to offline mode
helmfile -f helmfile.yaml.gotmpl -l name=ms-inference-scheduling apply \
  --args '--values override_ms_gptoss120b_offline_v050.yaml --values override_ms_gptoss120b_model.yaml --set decode.replicas=8 --set decode.parallelism.tensor=1'
```

The deployment will perform a rolling update to apply the new configuration.

## How the Helmfile Selector Works

The helmfile defines three releases with distinct names:
1. `infra-inference-scheduling` - Istio gateway and infrastructure
2. `gaie-inference-scheduling` - Gateway API Inference Extension (InferencePool)
3. `ms-inference-scheduling` - Model Service (vLLM deployment)

When you use `-l name=ms-inference-scheduling`, helmfile filters to **only** operate on that release. The `--args` flag passes the `--values` and `--set` arguments to the helm command for that specific release only.

**Example**: This command:
```bash
helmfile -f helmfile.yaml.gotmpl -l name=ms-inference-scheduling apply \
  --args '--values override_ms_gptoss120b_server_v050.yaml --values override_ms_gptoss120b_model.yaml --set decode.replicas=8'
```

Is equivalent to running:
```bash
helm upgrade ms-inference-scheduling llm-d-modelservice/llm-d-modelservice \
  --namespace llm-d-bench \
  --values ms-inference-scheduling/values.yaml \
  --values override_ms_gptoss120b_server_v050.yaml \
  --values override_ms_gptoss120b_model.yaml \
  --set decode.replicas=8 \
  --set decode.parallelism.tensor=1
```

The label selector ensures your overrides only affect the model service, not the infrastructure or GAIE components.

## Full vLLM Arguments Reference

### Server Mode
```bash
--disable-uvicorn-access-log
--gpu-memory-utilization=0.95
--no-enable-prefix-caching
--max-num-batched-tokens 2048
--max-num-seqs 448
--compilation-config {"max_cudagraph_capture_size": 2048}
--async-scheduling
```

### Offline Mode
```bash
--disable-uvicorn-access-log
--gpu-memory-utilization=0.95
--no-enable-prefix-caching
--max-num-batched-tokens 4096
--max-num-seqs 448
--compilation-config {"max_cudagraph_capture_size": 8192}
--async-scheduling
```

## Troubleshooting

### Pods Stuck in Pending

**Symptom**: Some decode pods remain in Pending state

**Possible Causes**:
1. Insufficient CPUs: Check node capacity with `kubectl describe nodes`
2. Insufficient GPUs: Verify 8 GPUs available with `kubectl get nodes -o json | jq '.items[].status.allocatable'`
3. Resource fragmentation: Try reducing `decode_replicas` or increasing `decode_cpu_request`

**Solution**:
```bash
# Check node resources
kubectl describe nodes | grep -A 10 "Allocated resources"

# Check pending pod events
kubectl describe pod -n llm-d-bench <pending-pod-name>
```

### CUDA Error 803

**Symptom**: Pods crash with "CUDA Error 803: system has unsupported display driver / cuda driver combination"

**Cause**: CUDA compatibility library mismatch

**Solution**: The override files already include the fix (hide-cuda-compat volume). Verify it's applied:
```bash
kubectl get pod -n llm-d-bench <pod-name> -o yaml | grep -A 5 "hide-cuda-compat"
```

### Model Not Loading

**Symptom**: Pods fail to start, logs show "Model not found" or PVC mounting issues

**Solution**: Verify model artifacts PVC and path:
```bash
# Check PVC exists
kubectl get pvc -n llm-d-bench models-storage

# Check PVC contents (if accessible)
kubectl exec -n llm-d-bench <any-pod> -- ls -la /model-cache/models/openai-gpt-oss-120b
```

## Contact & Support

For issues with these override files or deployment questions, refer to:
- llm-d repository: https://github.com/llm-d/llm-d
- vllm documentation: https://docs.vllm.ai

## Generated By

These override files were auto-generated from the vllmbench repository configurations:
- Server: `conf/gptoss120b_no_prefix_cache_v050.yaml`
- Offline: `conf/gptoss120b_no_prefix_cache_offline_v050.yaml`

Generated using `install_llmd.py --dump-overrides-only` feature.
