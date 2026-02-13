# EPP Configuration Comparison

## Overview

This document compares the default EPP (Endpoint Picker Plugin) configuration with the custom MLPerf-optimized configuration.

## Key Differences

### 1. Log Verbosity

**Default:**
```yaml
flags:
  kv-cache-usage-percentage-metric: "vllm:kv_cache_usage_perc"
  # No explicit verbosity setting
```

**Custom (epp_config.yaml):**
```yaml
flags:
  kv-cache-usage-percentage-metric: "vllm:kv_cache_usage_perc"
  v: 7  # Higher log verbosity for debugging
```

### 2. Plugins Configuration File

**Default:**
```yaml
pluginsConfigFile: "default-plugins.yaml"
# Uses built-in default configuration
```

**Custom (epp_config.yaml):**
```yaml
pluginsConfigFile: "mlperf-epp-config.yaml"
# Uses custom configuration defined inline
```

### 3. Custom Plugins Configuration

**Default:**
- No custom plugins configuration
- Uses built-in defaults from the GAIE image

**Custom (epp_config.yaml):**
```yaml
pluginsCustomConfig:
  mlperf-epp-config.yaml: |
    apiVersion: inference.networking.x-k8s.io/v1alpha1
    kind: EndpointPickerConfig
    plugins:
      - type: single-profile-handler
      - type: kv-cache-utilization-scorer
      - type: queue-scorer
      #- type: active-request-scorer  # Commented out
      - type: max-score-picker
    schedulingProfiles:
      - name: default
        plugins:
          - pluginRef: kv-cache-utilization-scorer
            weight: 2.0
          - pluginRef: queue-scorer
            weight: 2.0
          #- pluginRef: active-request-scorer
          #  weight: 2.0
          - pluginRef: max-score-picker
```

## Plugin Descriptions

### Enabled Plugins

1. **single-profile-handler**: Handles single scheduling profile
2. **kv-cache-utilization-scorer** (weight: 2.0): Scores endpoints based on KV cache utilization
3. **queue-scorer** (weight: 2.0): Scores endpoints based on queue depth
4. **max-score-picker**: Selects endpoint with maximum score

### Disabled Plugins

1. **active-request-scorer**: Commented out (not used in this configuration)

## Scoring Weights

The custom configuration uses equal weights for the two active scorers:
- **kv-cache-utilization-scorer**: 2.0
- **queue-scorer**: 2.0

This balances between:
- Choosing endpoints with more available KV cache
- Choosing endpoints with shorter queues

## Usage

To apply the custom EPP configuration:

```bash
./update_epp_config.sh [namespace]
```

Default namespace: `llm-d-bench`

## What the Update Script Does

1. Copies `epp_config.yaml` to the llm-d work directory
2. Applies the configuration using helmfile
3. Deletes the current EPP pod
4. Waits for the new EPP pod to start with the updated configuration
5. Displays the new pod status

## Verification

After applying the configuration, verify the EPP is using the new settings:

```bash
# Check EPP pod logs
kubectl logs -n llm-d-bench -l app.kubernetes.io/name=gaie-inference-scheduling-epp --tail=50

# Verify configuration is loaded
kubectl logs -n llm-d-bench -l app.kubernetes.io/name=gaie-inference-scheduling-epp | grep -i "plugin\|config"
```

## Reverting to Default

To revert to the default configuration:

1. Edit `epp_config.yaml` or create a new file with:
   ```yaml
   inferenceExtension:
     pluginsConfigFile: "default-plugins.yaml"
   ```

2. Remove the `pluginsCustomConfig` section

3. Run the update script:
   ```bash
   ./update_epp_config.sh
   ```
