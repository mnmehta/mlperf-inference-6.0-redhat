#!/bin/bash

# Deploy GPT-OSS-120B with llm-d v0.5.0
# Usage: ./deploy_gptoss120b_v050.sh [server|offline] [--dry-run]

set -e

# Configuration
LLMD_REPO="https://github.com/llm-d/llm-d"
LLMD_TAG="v0.5.0"
LLMD_DIR="${LLMD_DIR:-/tmp/llm-d}"
NAMESPACE="${NAMESPACE:-llm-d-bench}"
RELEASE_POSTFIX="${RELEASE_NAME_POSTFIX:-inference-scheduling}"
WORK_DIR="${LLMD_DIR}/guides/inference-scheduling"

# Parse arguments
MODE="${1:-}"
DRY_RUN=false

if [[ "$2" == "--dry-run" || "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    if [[ "$1" == "--dry-run" ]]; then
        MODE="${2:-}"
    fi
fi

if [[ "$MODE" != "server" && "$MODE" != "offline" ]]; then
    echo "ERROR: Invalid mode. Usage: $0 [server|offline] [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 server              # Deploy with server mode configuration"
    echo "  $0 offline             # Deploy with offline mode configuration"
    echo "  $0 server --dry-run    # Show what would be deployed without actually deploying"
    exit 1
fi

echo "=========================================="
echo "GPT-OSS-120B v0.5.0 Deployment"
echo "=========================================="
echo "Mode: ${MODE}"
echo "Dry Run: ${DRY_RUN}"
echo "Namespace: ${NAMESPACE}"
echo "Release Postfix: ${RELEASE_POSTFIX}"
echo "LLM-D Directory: ${LLMD_DIR}"
echo "=========================================="
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN MODE - No changes will be made"
    echo ""
fi

# Get script directory to find override files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check that override files exist
OVERRIDE_MODEL="${SCRIPT_DIR}/override_ms_gptoss120b_model.yaml"
if [[ "$MODE" == "server" ]]; then
    OVERRIDE_CONFIG="${SCRIPT_DIR}/override_ms_gptoss120b_server_v050.yaml"
else
    OVERRIDE_CONFIG="${SCRIPT_DIR}/override_ms_gptoss120b_offline_v050.yaml"
fi

if [[ ! -f "$OVERRIDE_MODEL" ]]; then
    echo "ERROR: Model override file not found: $OVERRIDE_MODEL"
    exit 1
fi

if [[ ! -f "$OVERRIDE_CONFIG" ]]; then
    echo "ERROR: Config override file not found: $OVERRIDE_CONFIG"
    exit 1
fi

echo "Using override files:"
echo "  Model:  $OVERRIDE_MODEL"
echo "  Config: $OVERRIDE_CONFIG"
echo ""

# Clone/update llm-d repository
if [[ ! -d "$LLMD_DIR" ]]; then
    echo "Cloning llm-d repository..."
    git clone "$LLMD_REPO" "$LLMD_DIR"
else
    echo "llm-d repository already exists at $LLMD_DIR"
fi

# Checkout the specific tag
echo "Checking out tag: $LLMD_TAG"
cd "$LLMD_DIR"
git fetch --tags --force
git checkout "$LLMD_TAG"
echo ""

# Copy override files to work directory
echo "Copying override files to ${WORK_DIR}..."
cp "$OVERRIDE_MODEL" "$WORK_DIR/"
cp "$OVERRIDE_CONFIG" "$WORK_DIR/"
echo ""

# Change to work directory
cd "$WORK_DIR"

# Export release name postfix for helmfile
export RELEASE_NAME_POSTFIX="$RELEASE_POSTFIX"

# Add helm repos
echo "Adding Helm repositories..."
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would run: helmfile repos"
else
    helmfile repos
fi
echo ""

# Cleanup old installations
echo "=========================================="
echo "Cleaning up old installations"
echo "=========================================="
echo "Deleting existing releases if they exist..."

# Delete releases in reverse order (ms, gaie, infra)
for release in "ms-${RELEASE_POSTFIX}" "gaie-${RELEASE_POSTFIX}" "infra-${RELEASE_POSTFIX}"; do
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would run: helm delete $release -n $NAMESPACE --ignore-not-found"
    else
        helm delete "$release" -n "$NAMESPACE" --ignore-not-found || true
    fi
done

echo "Cleanup complete (PVCs preserved)"
echo ""

# Ensure namespace has monitoring label and RBAC for Prometheus scraping
echo "=========================================="
echo "Configuring namespace for monitoring"
echo "=========================================="
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would ensure namespace $NAMESPACE has label: openshift.io/cluster-monitoring=true"
    echo "[DRY RUN] Would create RBAC permissions for Prometheus service account"
else
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true

    # Label namespace for Prometheus monitoring
    kubectl label namespace "$NAMESPACE" openshift.io/cluster-monitoring=true --overwrite
    echo "Namespace $NAMESPACE labeled for Prometheus monitoring"

    # Create RBAC permissions for Prometheus to scrape this namespace
    cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: prometheus-k8s
  namespace: $NAMESPACE
rules:
- apiGroups: [""]
  resources:
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources:
  - configmaps
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prometheus-k8s
  namespace: $NAMESPACE
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: prometheus-k8s
subjects:
- kind: ServiceAccount
  name: prometheus-k8s
  namespace: openshift-monitoring
EOF
    echo "RBAC permissions created for Prometheus service account"
fi
echo ""

# Deploy infrastructure (first time or if it doesn't exist)
echo "=========================================="
echo "Step 1/3: Deploying Infrastructure"
echo "=========================================="
INFRA_CMD="helmfile -f helmfile.yaml.gotmpl -l name=infra-${RELEASE_POSTFIX} apply -n $NAMESPACE"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would run: $INFRA_CMD"
else
    $INFRA_CMD
fi
echo ""

# Deploy GAIE (Gateway API Inference Extension)
echo "=========================================="
echo "Step 2/3: Deploying GAIE"
echo "=========================================="
GAIE_CMD="helmfile -f helmfile.yaml.gotmpl -l name=gaie-${RELEASE_POSTFIX} apply -n $NAMESPACE"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would run: $GAIE_CMD"
else
    $GAIE_CMD
fi
echo ""

# Deploy Model Service with override files
echo "=========================================="
echo "Step 3/3: Deploying Model Service (${MODE} mode)"
echo "=========================================="

OVERRIDE_MODEL_BASENAME="$(basename "$OVERRIDE_MODEL")"
OVERRIDE_CONFIG_BASENAME="$(basename "$OVERRIDE_CONFIG")"

MS_CMD="helmfile -f helmfile.yaml.gotmpl -l name=ms-${RELEASE_POSTFIX} apply -n $NAMESPACE --args \"--values ${OVERRIDE_CONFIG_BASENAME} --values ${OVERRIDE_MODEL_BASENAME} --set decode.replicas=8 --set decode.parallelism.tensor=1\""
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would run: $MS_CMD"
else
    helmfile -f helmfile.yaml.gotmpl -l name=ms-${RELEASE_POSTFIX} apply -n "$NAMESPACE" \
        --args "--values ${OVERRIDE_CONFIG_BASENAME} --values ${OVERRIDE_MODEL_BASENAME} --set decode.replicas=8 --set decode.parallelism.tensor=1"
fi
echo ""

# Wait for pods to be ready
echo "=========================================="
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry Run Complete!"
else
    echo "Deployment Complete!"
fi
echo "=========================================="
echo ""

if [[ "$DRY_RUN" == "false" ]]; then
    echo "Checking pod status..."
    kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/instance=ms-${RELEASE_POSTFIX}" || true
    echo ""
fi
echo "To watch pod status:"
echo "  kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=ms-${RELEASE_POSTFIX} -w"
echo ""
echo "To check logs:"
echo "  kubectl logs -n $NAMESPACE -l app.kubernetes.io/component=decode --tail=50 -f"
echo ""
echo "Gateway URL (from within cluster):"
echo "  http://infra-${RELEASE_POSTFIX}-inference-gateway-istio.${NAMESPACE}.svc.cluster.local:80"
echo ""
