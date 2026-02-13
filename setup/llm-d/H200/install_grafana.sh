#!/bin/bash
#
# Grafana Installation Script for OpenShift
#
# This script recreates the Grafana monitoring setup on a new OpenShift cluster.
# It installs the Grafana operator and configures:
# - Grafana instance
# - Prometheus datasource (with OpenShift monitoring integration)
# - LLM-D and inference gateway dashboards
# - Route for external access
#
# Usage: ./install_grafana.sh [namespace]
#   namespace: Target namespace (default: openshift-operators)
#

set -e

# Configuration
NAMESPACE="${1:-openshift-operators}"
GRAFANA_INSTANCE_NAME="grafana-a"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

wait_for_operator() {
    log_info "Waiting for Grafana operator to be ready..."
    local retries=0
    local max_retries=60

    while [ $retries -lt $max_retries ]; do
        if kubectl get deployment grafana-operator-controller-manager-v5 -n $NAMESPACE &>/dev/null; then
            if kubectl wait --for=condition=Available deployment/grafana-operator-controller-manager-v5 -n $NAMESPACE --timeout=10s &>/dev/null; then
                log_info "Grafana operator is ready"
                return 0
            fi
        fi
        sleep 5
        retries=$((retries + 1))
    done

    log_error "Timeout waiting for Grafana operator"
    return 1
}

wait_for_grafana() {
    log_info "Waiting for Grafana instance to be ready..."
    local retries=0
    local max_retries=60

    while [ $retries -lt $max_retries ]; do
        local stage=$(kubectl get grafana $GRAFANA_INSTANCE_NAME -n $NAMESPACE -o jsonpath='{.status.stage}' 2>/dev/null || echo "")
        if [ "$stage" = "complete" ]; then
            log_info "Grafana instance is ready"
            return 0
        fi
        sleep 5
        retries=$((retries + 1))
    done

    log_error "Timeout waiting for Grafana instance"
    return 1
}

# Step 1: Install Grafana Operator
install_operator() {
    log_info "Installing Grafana Operator..."

    cat <<EOF | kubectl apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: grafana-operator
  namespace: $NAMESPACE
spec:
  channel: v5
  installPlanApproval: Automatic
  name: grafana-operator
  source: community-operators
  sourceNamespace: openshift-marketplace
  startingCSV: grafana-operator.v5.21.2
EOF

    wait_for_operator
}

# Step 2: Create Service Account for Prometheus datasource
create_service_account() {
    log_info "Creating service account for Prometheus datasource..."

    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: grafana-datasource
  namespace: $NAMESPACE
EOF
}

# Step 3: Create ClusterRoleBinding for monitoring access
create_monitoring_access() {
    log_info "Creating ClusterRoleBinding for monitoring access..."

    cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: grafana-datasource-monitoring
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-monitoring-view
subjects:
- kind: ServiceAccount
  name: grafana-datasource
  namespace: $NAMESPACE
EOF
}

# Step 4: Create Grafana Instance
create_grafana_instance() {
    log_info "Creating Grafana instance..."

    cat <<EOF | kubectl apply -f -
apiVersion: grafana.integreatly.org/v1beta1
kind: Grafana
metadata:
  name: $GRAFANA_INSTANCE_NAME
  namespace: $NAMESPACE
  labels:
    dashboards: $GRAFANA_INSTANCE_NAME
    folders: $GRAFANA_INSTANCE_NAME
spec:
  config:
    auth:
      disable_login_form: "false"
    log:
      mode: console
    security:
      admin_password: start
      admin_user: root
  version: docker.io/grafana/grafana@sha256:70d9599b186ce287be0d2c5ba9a78acb2e86c1a68c9c41449454d0fc3eeb84e8
EOF

    wait_for_grafana
}

# Step 5: Get service account token for Prometheus datasource
get_sa_token() {
    log_info "Retrieving service account token..." >&2

    # Check if token secret already exists
    if ! kubectl get secret grafana-datasource-token -n $NAMESPACE &>/dev/null; then
        log_info "Creating service account token secret..." >&2
        # Create a token secret for the service account (long-lived)
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: grafana-datasource-token
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/service-account.name: grafana-datasource
type: kubernetes.io/service-account-token
EOF

        # Wait for token to be populated (only on first creation)
        log_info "Waiting for token to be populated..." >&2
        sleep 3
    else
        log_info "Token secret already exists, reusing existing token" >&2
    fi

    # Get token and trim any trailing newlines/whitespace
    local token=$(kubectl get secret grafana-datasource-token -n $NAMESPACE -o jsonpath='{.data.token}' | base64 -d | tr -d '\n\r')

    if [ -z "$token" ]; then
        log_error "Failed to retrieve token from secret" >&2
        return 1
    fi

    echo "$token"
}

# Step 6: Create Prometheus Datasource
create_prometheus_datasource() {
    # Check if datasource already exists
    if kubectl get grafanadatasource prometheus-datasource -n $NAMESPACE &>/dev/null; then
        log_warn "Prometheus datasource already exists. Updating configuration..."
    else
        log_info "Creating Prometheus datasource..."
    fi

    # Get the Thanos querier URL for the cluster
    local thanos_url
    thanos_url=$(kubectl get route thanos-querier -n openshift-monitoring -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

    if [ -z "$thanos_url" ]; then
        log_warn "Could not find thanos-querier route. Using default URL pattern."
        local cluster_domain=$(kubectl get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}' 2>/dev/null || echo "apps.example.com")
        thanos_url="thanos-querier-openshift-monitoring.${cluster_domain}"
    fi

    local sa_token=$(get_sa_token)

    if [ -z "$sa_token" ]; then
        log_error "Failed to retrieve service account token"
        return 1
    fi

    log_info "Using Thanos URL: https://${thanos_url}"

    # Create a temporary file to avoid shell interpolation issues with special characters
    local temp_file=$(mktemp)
    trap "rm -f $temp_file" EXIT

    cat > "$temp_file" <<EOF
apiVersion: grafana.integreatly.org/v1beta1
kind: GrafanaDatasource
metadata:
  name: prometheus-datasource
  namespace: $NAMESPACE
spec:
  allowCrossNamespaceImport: false
  datasource:
    access: proxy
    editable: false
    isDefault: true
    jsonData:
      httpHeaderName1: Authorization
      timeInterval: 30s
      tlsSkipVerify: true
    name: Prometheus (OpenShift Monitoring)
    secureJsonData:
      httpHeaderValue1: "Bearer ${sa_token}"
    type: prometheus
    url: "https://${thanos_url}"
  instanceSelector:
    matchLabels:
      dashboards: $GRAFANA_INSTANCE_NAME
EOF

    kubectl apply -f "$temp_file"
    rm -f "$temp_file"
}

# Step 7: Create Grafana Dashboards
create_dashboards() {
    log_info "Creating Grafana dashboards..."

    # Find all dashboard files in the script directory
    local dashboard_files=("$SCRIPT_DIR"/grafana-dashboard-*.yaml)

    if [ ! -e "${dashboard_files[0]}" ]; then
        log_warn "No dashboard files found matching pattern: grafana-dashboard-*.yaml"
        log_info "Skipping dashboard creation"
        return 0
    fi

    for dashboard_file in "${dashboard_files[@]}"; do
        if [ -f "$dashboard_file" ]; then
            local dashboard_name=$(basename "$dashboard_file" .yaml)
            dashboard_name=${dashboard_name#grafana-dashboard-}
            log_info "Creating dashboard: $dashboard_name"
            kubectl apply -f "$dashboard_file"
        fi
    done
}

# Step 8: Create Route for external access
create_route() {
    if kubectl get route grafana-route -n $NAMESPACE &>/dev/null; then
        log_warn "Route already exists. Updating configuration..."
    else
        log_info "Creating Route for Grafana access..."
    fi

    cat <<EOF | kubectl apply -f -
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: grafana-route
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/managed-by: grafana-operator
    dashboards: $GRAFANA_INSTANCE_NAME
    folders: $GRAFANA_INSTANCE_NAME
spec:
  port:
    targetPort: 3000
  tls:
    termination: edge
  to:
    kind: Service
    name: ${GRAFANA_INSTANCE_NAME}-service
    weight: 100
  wildcardPolicy: None
EOF

    # Get the route URL
    sleep 2
    local route_url=$(kubectl get route grafana-route -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

    if [ -n "$route_url" ]; then
        log_info "Grafana is accessible at: https://${route_url}"
        log_info "Default credentials: admin/start"
    fi
}

# Main installation flow
main() {
    log_info "========================================="
    log_info "Grafana Installation for OpenShift"
    log_info "========================================="
    log_info "Target namespace: $NAMESPACE"
    log_info ""

    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &>/dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi

    # Check if Grafana operator already installed
    if kubectl get subscription grafana-operator -n $NAMESPACE &>/dev/null; then
        log_warn "Grafana operator subscription already exists. Skipping installation."
    else
        install_operator
    fi

    # Create service account
    create_service_account

    # Create monitoring access
    create_monitoring_access

    # Create Grafana instance
    if kubectl get grafana $GRAFANA_INSTANCE_NAME -n $NAMESPACE &>/dev/null; then
        log_warn "Grafana instance already exists. Skipping creation."
    else
        create_grafana_instance
    fi

    # Create Prometheus datasource
    create_prometheus_datasource

    # Create dashboards
    create_dashboards

    # Create route
    create_route

    log_info ""
    log_info "========================================="
    log_info "Installation Complete!"
    log_info "========================================="
    log_info ""
    log_info "Next steps:"
    log_info "1. Access Grafana at the route URL shown above"
    log_info "2. Login with username: root, password: start"
    log_info "3. Navigate to Dashboards to view:"
    log_info "   - Inference Gateway Dashboard"
    log_info "   - LLM-D vLLM Overview"
    log_info "   - LLM-D Diagnostic Drilldown"
    log_info "   - LLM-D Failure Saturation"
    log_info ""
}

# Run main function
main
