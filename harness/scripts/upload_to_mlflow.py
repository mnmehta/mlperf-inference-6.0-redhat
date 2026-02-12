#!/usr/bin/env python3
"""
MLflow Artifact Upload Script
=============================
Script to upload directories to MLflow runs.

Usage:
    python3 upload_to_mlflow.py --experiment <experiment_name> --dir <directory> [--run-id <run_id>] [--tag <tag>]
    python3 upload_to_mlflow.py --experiment my_experiment --dir results --run-id abc123 --tag submission
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("ERROR: mlflow is not installed. Please install it with: pip install mlflow")
    sys.exit(1)


def get_or_create_experiment(client: MlflowClient, experiment_name: str):
    """Get existing experiment or create a new one."""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Creating new experiment: {experiment_name}")
            experiment_id = client.create_experiment(experiment_name)
            experiment = client.get_experiment(experiment_id)
        else:
            print(f"Using existing experiment: {experiment_name}")
        return experiment.experiment_id
    except Exception as e:
        print(f"ERROR: Failed to get or create experiment: {e}")
        sys.exit(1)


def get_or_create_run(client: MlflowClient, experiment_id: str, run_id: str = None):
    """Get existing run or create a new one."""
    if run_id:
        try:
            run = client.get_run(run_id)
            print(f"Using existing run: {run_id}")
            print(f"  Run name: {run.info.run_name or run_id}")
            return run_id
        except Exception as e:
            print(f"ERROR: Could not find run with ID: {run_id}")
            print(f"       Error: {e}")
            sys.exit(1)
    else:
        # Create a new run
        print("Creating new run...")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        print(f"Created new run: {run_id}")
        return run_id


def upload_directory(dir_path: str, experiment_name: str, run_id: str = None, 
                    artifact_path: str = None, tags: dict = None):
    """Upload a directory to MLflow."""
    directory = Path(dir_path)
    
    if not directory.exists():
        print(f"ERROR: Directory not found: {dir_path}")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"ERROR: Path is not a directory: {dir_path}")
        sys.exit(1)
    
    # Set artifact path
    if artifact_path is None:
        artifact_path = directory.name  # Use directory name
    
    print(f"Uploading directory {dir_path} to MLflow...")
    print(f"  Experiment: {experiment_name}")
    if run_id:
        print(f"  Run ID: {run_id}")
    print(f"  Artifact Path: {artifact_path}")
    if tags:
        print(f"  Tags: {tags}")
    print()
    
    try:
        # Get MLflow client
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"Using MLflow tracking URI: {tracking_uri}")
        else:
            print("Using default MLflow tracking URI (or local)")
        
        client = MlflowClient()
        
        # Get or create experiment
        experiment_id = get_or_create_experiment(client, experiment_name)
        
        # Get or create run
        actual_run_id = get_or_create_run(client, experiment_id, run_id)
        
        # Set tags if provided
        if tags:
            print("Setting tags...")
            for key, value in tags.items():
                client.set_tag(actual_run_id, key, value)
                print(f"  Set tag: {key} = {value}")
        
        # Upload directory
        print(f"Uploading directory contents...")
        mlflow.log_artifacts(str(directory), artifact_path, run_id=actual_run_id)
        
        print(f"✓ Successfully uploaded {directory.name} to run {actual_run_id}")
        print(f"  Artifact path: {artifact_path}/")
        print(f"  Run ID: {actual_run_id}")
        
    except Exception as e:
        print(f"ERROR: Failed to upload directory: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def parse_tags(tag_string: str) -> dict:
    """Parse tag string into dictionary.
    
    Supports formats:
    - key1=value1,key2=value2
    - key1:value1,key2:value2
    """
    if not tag_string:
        return {}
    
    tags = {}
    for pair in tag_string.split(','):
        pair = pair.strip()
        if '=' in pair:
            key, value = pair.split('=', 1)
        elif ':' in pair:
            key, value = pair.split(':', 1)
        else:
            print(f"WARNING: Invalid tag format: {pair} (expected key=value or key:value)")
            continue
        tags[key.strip()] = value.strip()
    
    return tags


def main():
    parser = argparse.ArgumentParser(
        description='Upload directories to MLflow runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload directory to a new run in an experiment
  python3 upload_to_mlflow.py --experiment my_experiment --dir results

  # Upload directory to a specific run
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --run-id abc123def456

  # Upload with tags
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --tag submission=final,version=1.0

  # Upload with custom artifact path
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --artifact-path submission

  # Set MLflow tracking URI via environment variable
  export MLFLOW_TRACKING_URI=http://mlflow-server:5000
  python3 upload_to_mlflow.py --experiment my_experiment --dir results
        """
    )
    
    parser.add_argument('--experiment', required=True, help='MLflow experiment name')
    parser.add_argument('--dir', required=True, help='Path to directory to upload')
    parser.add_argument('--run-id', help='MLflow run ID (if not provided, a new run will be created)')
    parser.add_argument('--artifact-path', help='Artifact path in MLflow (default: directory name)')
    parser.add_argument('--tag', '--mlflow-tag', dest='tags', help='MLflow tags in format key1=value1,key2=value2 or key1:value1,key2:value2')
    parser.add_argument('--tracking-uri', help='MLflow tracking URI (or set MLFLOW_TRACKING_URI env var)')
    
    args = parser.parse_args()
    
    if args.tracking_uri:
        os.environ['MLFLOW_TRACKING_URI'] = args.tracking_uri
    
    # Parse tags
    tags = parse_tags(args.tags) if args.tags else {}
    
    upload_directory(args.dir, args.experiment, args.run_id, args.artifact_path, tags)


if __name__ == '__main__':
    main()
