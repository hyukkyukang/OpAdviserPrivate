#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Disk I/O prediction accuracy (ReadIO and WriteIO).
Loads test data and trained models, calculates average MAPE.
"""

import os
import sys
import json
import joblib
import numpy as np
from autotune.utils.resource_parser import parse_collection_data
from scripts.evaluate_resource_model import calculate_mape


def main():
    # Default paths
    model_path = 'resource_models/resource_predictor.joblib'
    data_path = 'resource_data/resource_data.json'
    knob_config = 'scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json'
    knob_num = 15
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using train_resource_model.py")
        sys.exit(1)
    
    # Check if test data exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please collect data first using collect_resource_data.py")
        sys.exit(1)
    
    # Load models
    try:
        model_data = joblib.load(model_path)
        model_read_io = model_data['model_read_io']
        model_write_io = model_data['model_write_io']
        workload_encoder = model_data.get('workload_encoder', {})
        unique_workloads = workload_encoder.get('unique_workloads', [])
        n_workload_features = workload_encoder.get('n_features', 0)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load test data
    try:
        X, Y_cpu, Y_read_io, Y_write_io, workload_info = parse_collection_data(
            data_path, knob_config, knob_num
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Encode workload features
    workload_features = np.zeros((len(workload_info), n_workload_features))
    for i, info in enumerate(workload_info):
        workload_name = info.get('workload_name', info.get('workload_type', 'sysbench'))
        if len(unique_workloads) > 0:
            if workload_name in unique_workloads:
                idx = unique_workloads.index(workload_name)
                workload_features[i, idx] = 1.0
            else:
                workload_features[i, 0] = 1.0
        
        threads = info.get('threads', 40)
        if n_workload_features > 0:
            workload_features[i, -1] = threads / 100.0
    
    # Combine features
    X_combined = np.hstack([X, workload_features])
    
    # Make predictions
    try:
        read_io_pred, _ = model_read_io.predict(X_combined)
        write_io_pred, _ = model_write_io.predict(X_combined)
        read_io_pred = read_io_pred.flatten()
        write_io_pred = write_io_pred.flatten()
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)
    
    # Calculate MAPE for both ReadIO and WriteIO
    mape_read_io = calculate_mape(Y_read_io, read_io_pred)
    mape_write_io = calculate_mape(Y_write_io, write_io_pred)
    
    # Average MAPE
    avg_mape = (mape_read_io + mape_write_io) / 2.0
    
    # Print result
    print(f"Acc: {avg_mape:.2f}%")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

