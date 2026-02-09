#!/usr/bin/env python3
"""
Convert Run Submission Output to MLPerf Submission Structure
===========================================================
Converts the output directory from run_submission.py to the MLPerf submission
directory structure (like sample_1 but with RedHat instead of AMD).

Usage:
    python3 convert_to_submission.py --input-dir <output_dir> --output-dir <submission_dir> \\
        --system-name <system_name> --model <model_name> [--division closed|open]
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional


class SubmissionConverter:
    """Convert run submission output to MLPerf submission structure."""
    
    def __init__(self, input_dir: str, output_dir: str, system_name: str, 
                 model_name: str, division: str = 'closed', debug: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.system_name = system_name
        self.model_name = model_name
        self.division = division
        self.organization = 'RedHat'
        self.debug = debug
        
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
    def convert(self):
        """Perform the conversion."""
        print("==========================================")
        print("Converting to MLPerf Submission Structure")
        print("==========================================")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"System name: {self.system_name}")
        print(f"Model name: {self.model_name}")
        print(f"Division: {self.division}")
        print(f"Organization: {self.organization}")
        print("==========================================")
        print()
        
        # Create base structure
        base_path = self.output_dir / self.division / self.organization
        results_path = base_path / 'results' / self.system_name / self.model_name
        src_path = base_path / 'src' / self.model_name
        systems_path = base_path / 'systems'
        docs_path = base_path / 'documentation'
        
        # Create directories
        results_path.mkdir(parents=True, exist_ok=True)
        src_path.mkdir(parents=True, exist_ok=True)
        systems_path.mkdir(parents=True, exist_ok=True)
        docs_path.mkdir(parents=True, exist_ok=True)
        
        # Convert scenarios
        for scenario_dir in self.input_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
            
            scenario_name = scenario_dir.name.capitalize()  # server -> Server, offline -> Offline
            if scenario_name not in ['Server', 'Offline', 'Interactive', 'SingleStream']:
                print(f"Skipping unknown scenario: {scenario_name}")
                continue
            
            print(f"Converting {scenario_name} scenario...")
            self._convert_scenario(scenario_dir, results_path / scenario_name)
        
        # Create placeholder files
        self._create_placeholder_files(src_path, systems_path, docs_path)
        
        print()
        print("✓ Conversion completed successfully!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Results: {results_path}")
        print(f"  Source: {src_path}")
        print(f"  Systems: {systems_path}")
        print(f"  Documentation: {docs_path}")
    
    def _convert_scenario(self, input_scenario_dir: Path, output_scenario_dir: Path):
        """Convert a single scenario directory."""
        output_scenario_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert accuracy directory
        input_accuracy = input_scenario_dir / 'accuracy'
        if input_accuracy.exists():
            output_accuracy = output_scenario_dir / 'accuracy'
            output_accuracy.mkdir(parents=True, exist_ok=True)
            
            # Copy files from mlperf subdirectory directly to accuracy output (no run_1 subdirectory)
            input_mlperf = input_accuracy / 'mlperf'
            if input_mlperf.exists():
                # Copy only files from mlperf subdirectory (no subdirectories)
                for item in input_mlperf.iterdir():
                    if item.is_file():
                        dest = output_accuracy / item.name
                        shutil.copy2(item, dest)
                        if self.debug:
                            print(f"    [DEBUG] Copied: {item} -> {dest}")
                print(f"  Copied accuracy data")
            else:
                print(f"  Warning: mlperf subdirectory not found in {input_accuracy}")
            
            # Copy accuracy.txt if it exists
            input_accuracy_txt = input_accuracy / 'accuracy.txt'
            if input_accuracy_txt.exists():
                output_accuracy_txt = output_scenario_dir / 'accuracy.txt'
                shutil.copy2(input_accuracy_txt, output_accuracy_txt)
                if self.debug:
                    print(f"    [DEBUG] Copied: {input_accuracy_txt} -> {output_accuracy_txt}")
                print(f"  Copied accuracy.txt to scenario output directory")
        
        # Convert performance directory
        input_performance = input_scenario_dir / 'performance'
        if input_performance.exists():
            # Create run_1 subdirectory and copy files from mlperf subdirectory
            input_mlperf = input_performance / 'mlperf'
            if input_mlperf.exists():
                output_performance = output_scenario_dir / 'performance' / 'run_1'
                output_performance.mkdir(parents=True, exist_ok=True)
                # Copy only files from mlperf subdirectory to run_1 (no subdirectories)
                for item in input_mlperf.iterdir():
                    if item.is_file():
                        dest = output_performance / item.name
                        shutil.copy2(item, dest)
                        if self.debug:
                            print(f"    [DEBUG] Copied: {item} -> {dest}")
                print(f"  Copied performance data to run_1")
            else:
                print(f"  Warning: mlperf subdirectory not found in {input_performance}")
        
        # Convert compliance directory
        input_compliance = input_scenario_dir / 'compliance'
        if input_compliance.exists():
            # Copy TEST07 and TEST09 folders as-is
            for test_dir in input_compliance.iterdir():
                if test_dir.is_dir() and test_dir.name.upper().startswith('TEST'):
                    test_name = test_dir.name.upper()  # test07 -> TEST07
                    output_test_dir = output_scenario_dir / test_name
                    # Copy entire directory as-is
                    self._copy_directory(test_dir, output_test_dir)
                    if self.debug:
                        print(f"    [DEBUG] Copied directory: {test_dir} -> {output_test_dir}")
                    print(f"  Copied compliance {test_name} (as-is)")
        
        # Copy other files (measurements.json, mlperf.conf, user.conf, README.md)
        for file_name in ['measurements.json', 'mlperf.conf', 'user.conf', 'README.md']:
            input_file = input_scenario_dir / file_name
            if input_file.exists():
                dest = output_scenario_dir / file_name
                shutil.copy2(input_file, dest)
                if self.debug:
                    print(f"    [DEBUG] Copied: {input_file} -> {dest}")
                print(f"  Copied {file_name}")
    
    def _convert_compliance_test(self, input_test_dir: Path, output_test_dir: Path, test_name: str):
        """Convert a compliance test directory."""
        output_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy mlperf directory if it exists
        input_mlperf = input_test_dir / 'mlperf'
        if input_mlperf.exists():
            output_mlperf = output_test_dir / 'mlperf'
            self._copy_directory(input_mlperf, output_mlperf)
            if self.debug:
                print(f"    [DEBUG] Copied directory: {input_mlperf} -> {output_mlperf}")
        
        # Look for accuracy directory in mlperf
        input_accuracy = input_test_dir / 'mlperf' / 'mlperf_log_accuracy.json'
        if input_accuracy.exists():
            output_accuracy_dir = output_test_dir / 'accuracy'
            output_accuracy_dir.mkdir(parents=True, exist_ok=True)
            dest = output_accuracy_dir / 'mlperf_log_accuracy.json'
            shutil.copy2(input_accuracy, dest)
            if self.debug:
                print(f"    [DEBUG] Copied: {input_accuracy} -> {dest}")
        
        # Copy test-specific verify files
        if test_name == 'TEST07':
            verify_file = input_test_dir / 'verify_accuracy.txt'
            if verify_file.exists():
                dest = output_test_dir / 'verify_accuracy.txt'
                shutil.copy2(verify_file, dest)
                if self.debug:
                    print(f"    [DEBUG] Copied: {verify_file} -> {dest}")
                print(f"    Copied verify_accuracy.txt")
        elif test_name == 'TEST09':
            verify_file = input_test_dir / 'verify_output_len.txt'
            if verify_file.exists():
                dest = output_test_dir / 'verify_output_len.txt'
                shutil.copy2(verify_file, dest)
                if self.debug:
                    print(f"    [DEBUG] Copied: {verify_file} -> {dest}")
                print(f"    Copied verify_output_len.txt")
    
    def _copy_directory(self, src: Path, dst: Path):
        """Copy a directory recursively."""
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    
    def _create_placeholder_files(self, src_path: Path, systems_path: Path, docs_path: Path):
        """Create placeholder files if they don't exist."""
        # Create README.md in src
        src_readme = src_path / 'README.md'
        if not src_readme.exists():
            src_readme.write_text(f"""# {self.model_name}

Model source code and implementation details.

## Overview

This directory contains the source code and implementation details for {self.model_name}.

## Files

- Implementation files
- Configuration files
- Build scripts
""")
            print(f"  Created placeholder: {src_path / 'README.md'}")
        
        # Create system JSON file
        system_json = systems_path / f'{self.system_name}.json'
        if not system_json.exists():
            system_json.write_text(f"""{{
    "accelerator_frequency": "",
    "accelerator_host_interconnect": "",
    "accelerator_interconnect": "",
    "accelerator_memory_capacity": "",
    "accelerator_memory_configuration": "",
    "accelerators_per_node": "",
    "framework": "",
    "host_memory_capacity": "",
    "host_processor_model_name": "",
    "host_processors_per_node": "",
    "host_storage_capacity": "",
    "host_storage_type": "",
    "hw_notes": "",
    "number_of_nodes": 1,
    "number_of_type_nics_installed": "",
    "operating_system": "",
    "other_software_stack": "",
    "sw_notes": "",
    "system_name": "{self.system_name}",
    "system_type": "datacenter"
}}
""")
            print(f"  Created placeholder: {system_json}")
        
        # Create documentation files
        docs_readme = docs_path / 'README.md'
        if not docs_readme.exists():
            docs_readme.write_text(f"""# {self.organization} Submission Documentation

## Overview

This submission contains results for {self.model_name} running on {self.system_name}.

## Contents

- Calibration documentation
- System configuration details
- Performance tuning notes
""")
            print(f"  Created placeholder: {docs_path / 'README.md'}")
        
        # Create calibration.md
        calibration_md = docs_path / 'calibration.md'
        if not calibration_md.exists():
            calibration_md.write_text(f"""# Calibration Documentation

## Calibration Process

Describe the calibration process for {self.model_name}.

## Calibration Dataset

- Dataset details
- Calibration methodology

## Results

- Calibration accuracy
- Performance impact
""")
            print(f"  Created placeholder: {docs_path / 'calibration.md'}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert run_submission.py output to MLPerf submission structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert output directory
  python3 convert_to_submission.py \\
      --input-dir ./harness_output \\
      --output-dir ./submission \\
      --system-name "8xH100_2xEPYC_9654" \\
      --model "gpt-oss-120b"

  # Convert with open division
  python3 convert_to_submission.py \\
      --input-dir ./harness_output \\
      --output-dir ./submission \\
      --system-name "8xH100_2xEPYC_9654" \\
      --model "gpt-oss-120b" \\
      --division open
        """
    )
    
    parser.add_argument('--input-dir', required=True, help='Input directory from run_submission.py')
    parser.add_argument('--output-dir', required=True, help='Output directory for submission structure')
    parser.add_argument('--system-name', required=True, help='System name (e.g., 8xH100_2xEPYC_9654)')
    parser.add_argument('--model', required=True, help='Model name (e.g., gpt-oss-120b)')
    parser.add_argument('--division', choices=['closed', 'open'], default='closed',
                       help='Division (default: closed)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to print detailed file copy information')
    
    args = parser.parse_args()
    
    try:
        converter = SubmissionConverter(
            args.input_dir,
            args.output_dir,
            args.system_name,
            args.model,
            args.division,
            args.debug
        )
        converter.convert()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
