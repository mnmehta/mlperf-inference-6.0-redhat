# Grafana Installation for OpenShift

This directory contains scripts and configuration files to recreate the Grafana monitoring setup on a new OpenShift cluster.

## Overview

The Grafana setup includes:

1. **Grafana Operator v5.21.2** - Manages Grafana instances and dashboards
2. **Grafana Instance** (grafana-a) - Running Grafana v12.3.0
3. **Prometheus Datasource** - Connected to OpenShift cluster monitoring (Thanos)
4. **4 Custom Dashboards**:
   - **Inference Gateway Dashboard** - Monitors inference gateway metrics
   - **LLM-D vLLM Overview** - High-level vLLM deployment metrics
   - **LLM-D Diagnostic Drilldown** - Detailed diagnostic metrics
   - **LLM-D Failure Saturation** - Failure and saturation analysis
5. **OpenShift Route** - External access to Grafana UI

## Files in This Directory

```
mlperf_installer/
├── install_grafana.sh                              # Main installation script
├── GRAFANA_INSTALLATION.md                         # This documentation
├── grafana-dashboard-inference-gateway.yaml        # Inference Gateway dashboard
├── grafana-dashboard-llm-d-diagnostic-drilldown.yaml  # Diagnostic Drilldown dashboard
├── grafana-dashboard-llm-d-failure-saturation.yaml    # Failure Saturation dashboard
└── grafana-dashboard-llm-d-vllm-overview.yaml         # vLLM Overview dashboard
```

## Prerequisites

Before running the installation:

1. **OpenShift cluster** (tested on OpenShift 4.x)
2. **Cluster admin access** (required for ClusterRoleBinding)
3. **oc CLI or kubectl** configured with cluster access
4. **OpenShift monitoring stack** enabled (default on OpenShift)

## Quick Start

### Option 1: Install to openshift-operators namespace (recommended)

```bash
cd /home/michey/llmd_aug2025/mlperf_installer
./install_grafana.sh
```

### Option 2: Install to custom namespace

```bash
./install_grafana.sh my-namespace
```

## What the Script Does

The installation script performs the following steps:

### 1. Install Grafana Operator

Creates an Operator subscription to install Grafana Operator v5.21.2 from the community-operators catalog.

```yaml
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: grafana-operator
  namespace: openshift-operators
spec:
  channel: v5
  name: grafana-operator
  source: community-operators
```

### 2. Create Service Account

Creates a service account `grafana-datasource` that will be used to authenticate with Prometheus/Thanos.

### 3. Grant Monitoring Access

Creates a ClusterRoleBinding to grant the service account `cluster-monitoring-view` permissions, allowing it to query OpenShift monitoring metrics.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: grafana-datasource-monitoring
roleRef:
  kind: ClusterRole
  name: cluster-monitoring-view
subjects:
- kind: ServiceAccount
  name: grafana-datasource
  namespace: openshift-operators
```

### 4. Create Grafana Instance

Deploys a Grafana instance with:
- **Version**: Grafana 12.3.0
- **Admin user**: root
- **Admin password**: start (CHANGE THIS IN PRODUCTION!)

### 5. Create Prometheus Datasource

Configures Grafana to query OpenShift's Thanos querier endpoint. The script:
- Automatically detects the Thanos querier route URL
- Generates a long-lived service account token
- Configures bearer token authentication

### 6. Import Dashboards

Imports 4 pre-configured dashboards for monitoring LLM-D and inference gateway deployments.

### 7. Create Route

Creates an OpenShift route with TLS edge termination for external access.

## Accessing Grafana

After installation, the script will display the Grafana URL:

```
Grafana is accessible at: https://grafana-route-openshift-operators.apps.<cluster-domain>
Default credentials: root/start
```

Or retrieve it manually:

```bash
oc get route grafana-route -n openshift-operators
```

## Default Credentials

- **Username**: `root`
- **Password**: `start`

**IMPORTANT**: Change these credentials immediately after first login in production environments!

## Dashboard Details

### 1. Inference Gateway Dashboard

Monitors the Gateway API Inference Extension (GAIE) metrics:
- Request rates and latencies
- Pool health and status
- Routing decisions
- Error rates

**Metrics source**: https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/pkg/epp/metrics

### 2. LLM-D vLLM Overview

High-level overview of vLLM deployments:
- Token generation rates
- Request throughput
- GPU utilization
- Model serving metrics

### 3. LLM-D Diagnostic Drilldown

Detailed diagnostic metrics for troubleshooting:
- Per-pod metrics
- Queue depths
- Cache hit rates
- Detailed latency breakdowns

### 4. LLM-D Failure Saturation

Failure and saturation analysis:
- Error rates by type
- Resource saturation indicators
- Request queue saturation
- System bottlenecks

## Troubleshooting

### Operator Installation Issues

If the operator fails to install:

```bash
# Check operator pod logs
kubectl logs -n openshift-operators deployment/grafana-operator-controller-manager-v5

# Check install plan status
kubectl get installplan -n openshift-operators
```

### Grafana Instance Not Ready

```bash
# Check Grafana instance status
kubectl get grafana grafana-a -n openshift-operators -o yaml

# Check Grafana pod logs
kubectl logs -n openshift-operators deployment/grafana-a-deployment
```

### Datasource Connection Issues

If Grafana cannot connect to Prometheus:

1. **Verify Thanos querier is accessible**:
   ```bash
   oc get route thanos-querier -n openshift-monitoring
   ```

2. **Test token authentication**:
   ```bash
   TOKEN=$(kubectl get secret grafana-datasource-token -n openshift-operators -o jsonpath='{.data.token}' | base64 -d)
   THANOS_URL=$(kubectl get route thanos-querier -n openshift-monitoring -o jsonpath='{.spec.host}')
   curl -H "Authorization: Bearer $TOKEN" -k "https://${THANOS_URL}/api/v1/query?query=up"
   ```

3. **Check ClusterRoleBinding**:
   ```bash
   kubectl get clusterrolebinding grafana-datasource-monitoring -o yaml
   ```

### Dashboard Not Loading

If dashboards don't appear in Grafana:

1. **Check GrafanaDashboard resources**:
   ```bash
   kubectl get grafanadashboard -n openshift-operators
   ```

2. **Verify dashboard status**:
   ```bash
   kubectl get grafanadashboard inference-gateway -n openshift-operators -o yaml
   ```

3. **Check Grafana operator logs**:
   ```bash
   kubectl logs -n openshift-operators deployment/grafana-operator-controller-manager-v5
   ```

## Customization

### Change Admin Password

Edit the Grafana instance spec:

```bash
kubectl edit grafana grafana-a -n openshift-operators
```

Update the password field:

```yaml
spec:
  config:
    security:
      admin_password: YOUR_SECURE_PASSWORD
```

### Add Additional Datasources

Create a new GrafanaDatasource resource:

```yaml
apiVersion: grafana.integreatly.org/v1beta1
kind: GrafanaDatasource
metadata:
  name: my-datasource
  namespace: openshift-operators
spec:
  datasource:
    name: My Datasource
    type: prometheus
    url: https://my-prometheus:9090
  instanceSelector:
    matchLabels:
      dashboards: grafana-a
```

### Import Additional Dashboards

To add more dashboards:

1. Export dashboard JSON from an existing Grafana instance
2. Create a GrafanaDashboard resource:

```yaml
apiVersion: grafana.integreatly.org/v1beta1
kind: GrafanaDashboard
metadata:
  name: my-dashboard
  namespace: openshift-operators
spec:
  instanceSelector:
    matchLabels:
      dashboards: grafana-a
  json: |
    {
      "dashboard": { ... }
    }
```

## Uninstallation

To remove the Grafana installation:

```bash
# Delete dashboards
kubectl delete grafanadashboard -n openshift-operators --all

# Delete datasource
kubectl delete grafanadatasource prometheus-datasource -n openshift-operators

# Delete route
kubectl delete route grafana-route -n openshift-operators

# Delete Grafana instance
kubectl delete grafana grafana-a -n openshift-operators

# Delete ClusterRoleBinding
kubectl delete clusterrolebinding grafana-datasource-monitoring

# Delete service account
kubectl delete serviceaccount grafana-datasource -n openshift-operators

# Uninstall operator (optional)
kubectl delete subscription grafana-operator -n openshift-operators
kubectl delete csv -n openshift-operators -l operators.coreos.com/grafana-operator.openshift-operators
```

## Security Considerations

### Production Deployment

For production use:

1. **Change default admin credentials** immediately
2. **Use RBAC** to restrict Grafana access:
   ```bash
   kubectl edit grafana grafana-a -n openshift-operators
   ```
   Add OAuth or LDAP configuration

3. **Rotate service account token** regularly:
   ```bash
   kubectl delete secret grafana-datasource-token -n openshift-operators
   # Script will recreate it on next run
   ```

4. **Enable audit logging** in Grafana config

5. **Use network policies** to restrict Grafana pod network access

### Service Account Token

The script creates a long-lived service account token (10-year expiry) stored in a Secret. This is acceptable for internal monitoring but consider:

- Regular token rotation policy
- Use short-lived tokens with automatic refresh if possible
- Monitor token usage via audit logs

## Migration Notes

When migrating from the source cluster (psap-aus-h200):

1. **Dashboard UIDs will change** - Update any external links
2. **Route hostname will change** - Based on new cluster's domain
3. **Thanos URL will change** - Script auto-detects the new URL
4. **Service account token must be regenerated** - Cannot be exported/imported

## Reference

### Original Cluster Configuration

This setup was exported from:
- **Cluster**: psap-aus-h200.ibm.rhperfscale.org
- **Namespace**: openshift-operators
- **Grafana Version**: 12.3.0
- **Operator Version**: v5.21.2
- **Export Date**: 2026-02-12

### Grafana Operator Documentation

- GitHub: https://github.com/grafana-operator/grafana-operator
- Operator Hub: https://operatorhub.io/operator/grafana-operator
- Documentation: https://grafana-operator.github.io/grafana-operator/

### OpenShift Monitoring

- Monitoring Stack: https://docs.openshift.com/container-platform/latest/monitoring/monitoring-overview.html
- Thanos Documentation: https://thanos.io/

## Support

For issues specific to:
- **Grafana Operator**: https://github.com/grafana-operator/grafana-operator/issues
- **LLM-D Dashboards**: Contact the LLM-D project team
- **OpenShift Monitoring**: Red Hat support or OpenShift documentation

## Changelog

### 2026-02-12
- Initial export from psap-aus-h200 cluster
- Created automated installation script
- Exported 4 LLM-D monitoring dashboards
- Documented full installation process

## Idempotency

The installation script is **fully idempotent** and can be safely run multiple times.

### What Happens on Repeated Runs

When you run `./install_grafana.sh` on a cluster where Grafana is already installed:

1. **Grafana Operator**: Skips installation if subscription already exists
2. **Service Account**: Updates if configuration changed, otherwise no-op
3. **ClusterRoleBinding**: Updates if configuration changed, otherwise no-op
4. **Grafana Instance**: Skips creation if instance already exists
5. **Token Secret**: Reuses existing token, doesn't regenerate
6. **Prometheus Datasource**: Updates configuration (e.g., if Thanos URL changed)
7. **Dashboards**: Updates dashboard definitions to latest version
8. **Route**: Updates if configuration changed, otherwise no-op

### Safe Operations

All resource creation uses `kubectl apply`, which is idempotent:
- Creates the resource if it doesn't exist
- Updates the resource if it exists and configuration changed
- No-op if it exists and configuration is unchanged

### When to Rerun

It's safe and recommended to rerun the script when:
- Updating dashboard definitions
- Recovering from partial installation failure
- Ensuring all components are correctly configured
- Migrating to a new Thanos endpoint

### What Doesn't Change

On reruns, these values are **preserved**:
- Service account token (reused, not regenerated)
- Grafana admin password (from existing instance)
- Dashboard customizations (if you edit dashboards in Grafana UI, they'll be overwritten)
- Route hostname (assigned by OpenShift, remains stable)

### Example: Safe Rerun

```bash
# First run - full installation
./install_grafana.sh

# Later - update dashboards or ensure configuration
./install_grafana.sh

# Output will show:
# [WARN] Grafana operator subscription already exists. Skipping installation.
# [WARN] Grafana instance already exists. Skipping creation.
# [INFO] Token secret already exists, reusing existing token
# [WARN] Prometheus datasource already exists. Updating configuration...
# [INFO] Creating dashboard: inference-gateway
# [WARN] Route already exists. Updating configuration...
```
