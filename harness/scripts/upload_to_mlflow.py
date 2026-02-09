#!/usr/bin/env python3
"""
MLflow Artifact Upload Script
=============================
Simple script to upload .zip or .xz files to MLflow.

Usage:
    python3 upload_to_mlflow.py --file <archive_file> --run-id <run_id> [--artifact-path <path>]
    python3 upload_to_mlflow.py --file results.zip --run-id abc123 --artifact-path submission
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import mlflow
except ImportError:
    print("ERROR: mlflow is not installed. Please install it with: pip install mlflow")
    sys.exit(1)


def upload_artifact(file_path: str, run_id: str, artifact_path: str = None):
    """Upload a file to MLflow."""
    file = Path(file_path)
    
    if not file.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    # Check if it's a zip or xz file
    if file.suffix not in ['.zip', '.xz', '.tar.xz']:
        print(f"WARNING: File extension is {file.suffix}, expected .zip or .xz")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Set artifact path
    if artifact_path is None:
        artifact_path = file.stem  # Use filename without extension
    
    print(f"Uploading {file_path} to MLflow...")
    print(f"  Run ID: {run_id}")
    print(f"  Artifact Path: {artifact_path}")
    print()
    
    try:
        # Get MLflow client
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"Using MLflow tracking URI: {tracking_uri}")
        else:
            print("Using default MLflow tracking URI (or local)")
        
        # Get the run
        client = mlflow.tracking.MlflowClient()
        
        # Verify run exists
        try:
            run = client.get_run(run_id)
            print(f"Found run: {run.info.run_name or run_id}")
        except Exception as e:
            print(f"ERROR: Could not find run with ID: {run_id}")
            print(f"       Error: {e}")
            sys.exit(1)
        
        # Log artifact
        print(f"Uploading {file.name}...")
        mlflow.log_artifact(file_path, artifact_path, run_id=run_id)
        
        print(f"✓ Successfully uploaded {file.name} to run {run_id}")
        print(f"  Artifact path: {artifact_path}/{file.name}")
        
    except Exception as e:
        print(f"ERROR: Failed to upload artifact: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Upload .zip or .xz files to MLflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a zip file
  python3 upload_to_mlflow.py --file results.zip --run-id abc123def456

  # Upload with custom artifact path
  python3 upload_to_mlflow.py --file submission.tar.xz --run-id abc123 --artifact-path submissions

  # Set MLflow tracking URI via environment variable
  export MLFLOW_TRACKING_URI=http://mlflow-server:5000
  python3 upload_to_mlflow.py --file results.zip --run-id abc123
        """
    )
    
    parser.add_argument('--file', required=True, help='Path to .zip or .xz file to upload')
    parser.add_argument('--run-id', required=True, help='MLflow run ID')
    parser.add_argument('--artifact-path', help='Artifact path in MLflow (default: filename without extension)')
    parser.add_argument('--tracking-uri', help='MLflow tracking URI (or set MLFLOW_TRACKING_URI env var)')
    
    args = parser.parse_args()
    
    if args.tracking_uri:
        os.environ['MLFLOW_TRACKING_URI'] = args.tracking_uri
    
    upload_artifact(args.file, args.run_id, args.artifact_path)


if __name__ == '__main__':
    main()
