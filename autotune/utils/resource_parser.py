"""
Parser for extracting resource data (CPU, ReadIO, WriteIO) from history files.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Tuple, Optional
from autotune.knobs import initialize_knobs, knobDF2action


RESOURCE_KEYS = ['cpu', 'readIO', 'writeIO', 'virtualMem', 'physicalMem', 'dirty', 'hit', 'data']


def _extract_resource_metrics(resource: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(resource, dict):
        try:
            return (
                float(resource.get('cpu', 0)),
                float(resource.get('readIO', 0)),
                float(resource.get('writeIO', 0)),
            )
        except (TypeError, ValueError):
            return None

    if isinstance(resource, np.ndarray):
        resource = resource.tolist()

    if isinstance(resource, (list, tuple)) and len(resource) >= 3:
        try:
            return float(resource[0]), float(resource[1]), float(resource[2])
        except (TypeError, ValueError):
            return None

    return None


def _extract_workload_info(sample: Dict[str, Any]) -> Dict[str, Any]:
    context = sample.get('context') or {}
    workload = sample.get('workload') or {}
    return {
        'workload_type': (
            workload.get('workload_type')
            or workload.get('type')
            or context.get('workload_type')
            or context.get('workload')
            or 'unknown'
        ),
        'workload_name': workload.get('workload_name') or workload.get('name') or context.get('workload_name') or 'unknown',
        'threads': workload.get('threads') or context.get('threads') or 0,
        'trial_state': sample.get('trial_state', 'UNKNOWN')
    }


def parse_resource_data_from_json(file_path: str, knob_config_file: str, knob_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Parse resource data from history JSON file.
    
    Parameters:
    -----------
    file_path : str
        Path to history JSON file
    knob_config_file : str
        Path to knob configuration file
    knob_num : int
        Number of knobs
    
    Returns:
    --------
    X : np.ndarray [n_samples, n_features]
        Configuration features (in action space)
    Y_cpu : np.ndarray [n_samples]
        CPU usage values
    Y_read_io : np.ndarray [n_samples]
        Read I/O values
    Y_write_io : np.ndarray [n_samples]
        Write I/O values
    workload_info : List[Dict]
        Workload metadata for each sample
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'data' not in data:
        raise ValueError(f"No 'data' key found in {file_path}")
    
    samples = data['data']
    
    # Initialize knobs
    knobs_detail = initialize_knobs(knob_config_file, knob_num)
    
    # Extract data
    configs_list = []
    cpu_list = []
    read_io_list = []
    write_io_list = []
    workload_info_list = []
    
    for sample in samples:
        # Extract configuration
        config_dict = sample.get('configuration', {})
        if not config_dict:
            continue
        
        resource_metrics = _extract_resource_metrics(sample.get('resource', []))
        if resource_metrics is None:
            continue

        cpu, read_io, write_io = resource_metrics
        
        # Filter invalid data
        if cpu <= 0 or read_io < 0 or write_io < 0:
            continue
        
        workload_info = _extract_workload_info(sample)
        
        configs_list.append(config_dict)
        cpu_list.append(cpu)
        read_io_list.append(read_io)
        write_io_list.append(write_io)
        workload_info_list.append(workload_info)
    
    if len(configs_list) == 0:
        raise ValueError(f"No valid samples found in {file_path}")
    
    # Convert configurations to DataFrame
    config_df = pd.DataFrame(configs_list)
    
    # Convert to action space (normalized 0-1)
    try:
        X = knobDF2action(config_df)
    except Exception as e:
        # Fallback: try to handle missing knobs
        print(f"Warning: Error converting to action space: {e}")
        # Filter to only knobs that exist in knobs_detail
        available_knobs = [k for k in config_df.columns if k in knobs_detail]
        config_df_filtered = config_df[available_knobs]
        X = knobDF2action(config_df_filtered)
    
    Y_cpu = np.array(cpu_list)
    Y_read_io = np.array(read_io_list)
    Y_write_io = np.array(write_io_list)
    
    return X, Y_cpu, Y_read_io, Y_write_io, workload_info_list


def parse_multiple_files(file_paths: List[str], knob_config_file: str, knob_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Parse resource data from multiple history JSON files.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of paths to history JSON files
    knob_config_file : str
        Path to knob configuration file
    knob_num : int
        Number of knobs
    
    Returns:
    --------
    X_all : np.ndarray [n_samples, n_features]
        All configuration features
    Y_cpu_all : np.ndarray [n_samples]
        All CPU usage values
    Y_read_io_all : np.ndarray [n_samples]
        All Read I/O values
    Y_write_io_all : np.ndarray [n_samples]
        All Write I/O values
    workload_info_all : List[Dict]
        All workload metadata
    """
    X_list = []
    Y_cpu_list = []
    Y_read_io_list = []
    Y_write_io_list = []
    workload_info_list = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        if os.path.getsize(file_path) == 0:
            print(f"Warning: Empty file: {file_path}")
            continue
        
        try:
            X, Y_cpu, Y_read_io, Y_write_io, workload_info = parse_resource_data_from_json(
                file_path, knob_config_file, knob_num
            )
            
            X_list.append(X)
            Y_cpu_list.extend(Y_cpu)
            Y_read_io_list.extend(Y_read_io)
            Y_write_io_list.extend(Y_write_io)
            workload_info_list.extend(workload_info)
            
            print(f"Parsed {len(Y_cpu)} samples from {file_path}")
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(X_list) == 0:
        raise ValueError("No valid data found in any files")
    
    # Stack all data
    X_all = np.vstack(X_list)
    Y_cpu_all = np.array(Y_cpu_list)
    Y_read_io_all = np.array(Y_read_io_list)
    Y_write_io_all = np.array(Y_write_io_list)
    
    return X_all, Y_cpu_all, Y_read_io_all, Y_write_io_all, workload_info_list


def parse_collection_data(file_path: str, knob_config_file: str, knob_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Parse data from the collection script output format.
    
    Parameters:
    -----------
    file_path : str
        Path to resource_data.json file
    knob_config_file : str
        Path to knob configuration file
    knob_num : int
        Number of knobs
    
    Returns:
    --------
    X : np.ndarray [n_samples, n_features]
        Configuration features
    Y_cpu : np.ndarray [n_samples]
        CPU usage values
    Y_read_io : np.ndarray [n_samples]
        Read I/O values
    Y_write_io : np.ndarray [n_samples]
        Write I/O values
    workload_info : List[Dict]
        Workload metadata
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'data' not in data:
        raise ValueError(f"No 'data' key found in {file_path}")
    
    samples = data['data']
    
    # Initialize knobs
    knobs_detail = initialize_knobs(knob_config_file, knob_num)
    
    # Extract data
    configs_list = []
    cpu_list = []
    read_io_list = []
    write_io_list = []
    workload_info_list = []
    
    for sample in samples:
        # Extract configuration
        config_dict = sample.get('configuration', {})
        if not config_dict:
            continue
        
        resource_metrics = _extract_resource_metrics(sample.get('resource', {}))
        if resource_metrics is None:
            continue

        cpu, read_io, write_io = resource_metrics
        
        # Filter invalid data
        if cpu <= 0 or read_io < 0 or write_io < 0:
            continue
        
        # Extract workload info
        workload = sample.get('workload') or {}
        workload_info = {
            'workload_type': workload.get('workload_type', workload.get('type', 'sbrw')),
            'threads': workload.get('threads', 40),
            'workload_name': workload.get('workload_name', workload.get('name', workload.get('type', 'sysbench')))
        }
        
        configs_list.append(config_dict)
        cpu_list.append(cpu)
        read_io_list.append(read_io)
        write_io_list.append(write_io)
        workload_info_list.append(workload_info)
    
    if len(configs_list) == 0:
        raise ValueError(f"No valid samples found in {file_path}")
    
    # Convert configurations to DataFrame
    config_df = pd.DataFrame(configs_list)
    
    # Convert to action space
    try:
        X = knobDF2action(config_df)
    except Exception as e:
        print(f"Warning: Error converting to action space: {e}")
        # Filter to only knobs that exist
        available_knobs = [k for k in config_df.columns if k in knobs_detail]
        config_df_filtered = config_df[available_knobs]
        X = knobDF2action(config_df_filtered)
    
    Y_cpu = np.array(cpu_list)
    Y_read_io = np.array(read_io_list)
    Y_write_io = np.array(write_io_list)
    
    return X, Y_cpu, Y_read_io, Y_write_io, workload_info_list
