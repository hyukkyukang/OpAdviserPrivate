#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for CPU/IO resource prediction models.
Trains Random Forest models to predict CPU usage and Disk I/O.
"""

import os
import sys
import argparse
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from autotune.utils.resource_parser import parse_collection_data, parse_multiple_files
from autotune.optimizer.surrogate.base.rf_with_instances import RandomForestWithInstances
from autotune.knobs import initialize_knobs
from autotune.tuner import DBTuner
from autotune.utils.config_space import ConfigurationSpace
from scripts.evaluate_resource_model import evaluate_all_models, calculate_mape
import time


def encode_workload_features(workload_info_list):
    """
    Encode workload information as features.
    
    Parameters:
    -----------
    workload_info_list : List[Dict]
        List of workload metadata dictionaries
    
    Returns:
    --------
    features : np.ndarray [n_samples, n_features]
        Encoded workload features
    """
    # Simple encoding: one-hot for workload type, numeric for threads
    unique_workloads = set()
    for info in workload_info_list:
        workload_name = info.get('workload_name', info.get('workload_type', 'unknown'))
        unique_workloads.add(workload_name)
    
    unique_workloads = sorted(list(unique_workloads))
    n_workloads = len(unique_workloads)
    
    features = np.zeros((len(workload_info_list), n_workloads + 1))
    
    for i, info in enumerate(workload_info_list):
        workload_name = info.get('workload_name', info.get('workload_type', 'unknown'))
        if workload_name in unique_workloads:
            idx = unique_workloads.index(workload_name)
            features[i, idx] = 1.0
        
        # Add thread count as feature (normalized)
        threads = info.get('threads', 40)
        features[i, -1] = threads / 100.0  # Normalize to 0-1 range
    
    return features, unique_workloads


def get_types_and_bounds(config_space, n_workload_features=0):
    """
    Extract types and bounds from ConfigurationSpace for RandomForest.
    
    Parameters:
    -----------
    config_space : ConfigurationSpace
        Configuration space object
    n_workload_features : int
        Number of workload context features to add
    
    Returns:
    --------
    types : np.ndarray
        Types array for RF
    bounds : list
        Bounds list for RF
    """
    types = []
    bounds = []
    
    for hp in config_space.get_hyperparameters():
        if hasattr(hp, 'choices'):  # Categorical
            types.append(len(hp.choices))
            bounds.append((len(hp.choices), np.nan))
        elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):  # Numeric
            types.append(0)  # Continuous
            bounds.append((hp.lower, hp.upper))
        else:
            types.append(0)
            bounds.append((0.0, 1.0))
    
    # Add workload features (all continuous)
    for _ in range(n_workload_features):
        types.append(0)
        bounds.append((0.0, 1.0))
    
    return np.array(types, dtype=np.uint), bounds


def train_model(X_train, y_train, types, bounds, workload_features_train=None, num_trees=100):
    """
    Train a Random Forest model for resource prediction.
    
    Parameters:
    -----------
    X_train : np.ndarray [n_samples, n_features]
        Training configurations
    y_train : np.ndarray [n_samples]
        Training target values
    types : np.ndarray
        Types array for RF
    bounds : list
        Bounds list for RF
    workload_features_train : np.ndarray, optional
        Workload context features
    num_trees : int
        Number of trees in RF
    
    Returns:
    --------
    model : RandomForestWithInstances
        Trained model
    """
    # Combine configuration and workload features
    if workload_features_train is not None:
        X_combined = np.hstack([X_train, workload_features_train])
    else:
        X_combined = X_train
    
    # Create and train model
    model = RandomForestWithInstances(
        types=types,
        bounds=bounds,
        log_y=False,  # CPU/I/O are not log-normal
        num_trees=num_trees,
        min_samples_split=3,
        min_samples_leaf=3,
        seed=42
    )
    
    print(f"  Training with {len(X_train)} samples, {X_combined.shape[1]} features...")
    model.train(X_combined, y_train.reshape(-1, 1))
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train resource prediction models')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to resource data JSON file')
    parser.add_argument('--knob_config', type=str, required=True,
                       help='Path to knob configuration file')
    parser.add_argument('--knob_num', type=int, required=True,
                       help='Number of knobs')
    parser.add_argument('--output_dir', type=str, default='resource_models',
                       help='Output directory for trained models')
    parser.add_argument('--num_trees', type=int, default=100,
                       help='Number of trees in Random Forest (default: 100)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size (default: 0.2)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Resource Prediction Model Training")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Knob config: {args.knob_config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of trees: {args.num_trees}")
    print("="*60)
    
    start_time = time.time()
    
    # Load data
    print("\n[1/6] Loading data...")
    try:
        X, Y_cpu, Y_read_io, Y_write_io, workload_info = parse_collection_data(
            args.data_file, args.knob_config, args.knob_num
        )
        print(f"  Loaded {len(X)} samples")
        print(f"  Features: {X.shape[1]}")
        print(f"  CPU range: [{Y_cpu.min():.2f}, {Y_cpu.max():.2f}]")
        print(f"  Read I/O range: [{Y_read_io.min():.2f}, {Y_read_io.max():.2f}]")
        print(f"  Write I/O range: [{Y_write_io.min():.2f}, {Y_write_io.max():.2f}]")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Encode workload features
    print("\n[2/6] Encoding workload features...")
    workload_features, unique_workloads = encode_workload_features(workload_info)
    print(f"  Workload types: {unique_workloads}")
    print(f"  Workload features shape: {workload_features.shape}")
    
    # Setup configuration space for types/bounds
    print("\n[3/6] Setting up configuration space...")
    # Create a dummy tuner to access setup_configuration_space
    from autotune.utils.config import parse_args as parse_config
    from autotune.database.mysqldb import MysqlDB
    from autotune.dbenv import DBEnv
    
    # Create minimal config for tuner
    config_content = f"""[database]
knob_config_file = {args.knob_config}
knob_num = {args.knob_num}
db = mysql
host = localhost
port = 3306
user = root
passwd = password
sock = /var/run/mysqld/mysqld.sock
cnf = scripts/template/experiment_normandy.cnf
mysqld = /usr/sbin/mysqld
workload = sysbench
workload_type = sbrw
thread_num = 40
workload_time = 30
workload_warmup_time = 5
online_mode = True
remote_mode = False
[tune]
task_id = dummy
performance_metric = ['tps']
reference_point = [None, None]
constraints = 
max_runs = 10
optimize_method = SMAC
space_transfer = False
auto_optimizer = False
acq_optimizer_type = random
transfer_framework = none
data_repo = repo
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        temp_config = f.name
    
    try:
        args_db, args_tune = parse_config(temp_config)
        db = MysqlDB(args_db)
        env = DBEnv(args_db, args_tune, db)
        tuner = DBTuner(args_db, args_tune, env)
        config_space = tuner.setup_configuration_space(args.knob_config, args.knob_num)
    finally:
        os.unlink(temp_config)
    
    n_workload_features = workload_features.shape[1]
    types, bounds = get_types_and_bounds(config_space, n_workload_features)
    print(f"  Types: {types}")
    print(f"  Total features (config + workload): {len(types)}")
    
    # Split data: train/val/test (60/20/20)
    print("\n[4/6] Splitting data...")
    # First split: train+val vs test
    X_temp, X_test, y_cpu_temp, y_cpu_test, y_read_io_temp, y_read_io_test, \
    y_write_io_temp, y_write_io_test, wf_temp, wf_test = train_test_split(
        X, Y_cpu, Y_read_io, Y_write_io, workload_features,
        test_size=args.test_size, random_state=42
    )
    
    # Second split: train vs val
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_cpu_train, y_cpu_val, y_read_io_train, y_read_io_val, \
    y_write_io_train, y_write_io_val, wf_train, wf_val = train_test_split(
        X_temp, y_cpu_temp, y_read_io_temp, y_write_io_temp, wf_temp,
        test_size=val_size_adjusted, random_state=42
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Train models
    print("\n[5/6] Training models...")
    
    print("  Training CPU model...")
    model_cpu = train_model(X_train, y_cpu_train, types, bounds, wf_train, args.num_trees)
    
    print("  Training Read I/O model...")
    model_read_io = train_model(X_train, y_read_io_train, types, bounds, wf_train, args.num_trees)
    
    print("  Training Write I/O model...")
    model_write_io = train_model(X_train, y_write_io_train, types, bounds, wf_train, args.num_trees)
    
    # Evaluate on validation set
    print("\n[6/6] Evaluating models...")
    
    # CPU predictions
    X_val_combined = np.hstack([X_val, wf_val])
    cpu_val_pred, _ = model_cpu.predict(X_val_combined)
    read_io_val_pred, _ = model_read_io.predict(X_val_combined)
    write_io_val_pred, _ = model_write_io.predict(X_val_combined)
    
    print("\nValidation Set Results:")
    val_metrics = evaluate_all_models(
        y_cpu_val, cpu_val_pred.flatten(),
        y_read_io_val, read_io_val_pred.flatten(),
        y_write_io_val, write_io_val_pred.flatten()
    )
    
    # Evaluate on test set
    X_test_combined = np.hstack([X_test, wf_test])
    cpu_test_pred, _ = model_cpu.predict(X_test_combined)
    read_io_test_pred, _ = model_read_io.predict(X_test_combined)
    write_io_test_pred, _ = model_write_io.predict(X_test_combined)
    
    print("\nTest Set Results:")
    test_metrics = evaluate_all_models(
        y_cpu_test, cpu_test_pred.flatten(),
        y_read_io_test, read_io_test_pred.flatten(),
        y_write_io_test, write_io_test_pred.flatten()
    )
    
    # Check if models meet the <10% MAPE requirement
    cpu_pass = test_metrics['cpu']['mape'] < 10
    read_io_pass = test_metrics['read_io']['mape'] < 10
    write_io_pass = test_metrics['write_io']['mape'] < 10
    all_pass = cpu_pass and read_io_pass and write_io_pass
    
    # Save models
    print("\nSaving models...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Note: RandomForestWithInstances models contain SWIG objects that can't be pickled.
    # Save training data instead so models can be retrained when loaded.
    model_data = {
        # Training data for retraining models
        'X_train': X_train,
        'y_cpu_train': y_cpu_train,
        'y_read_io_train': y_read_io_train,
        'y_write_io_train': y_write_io_train,
        'wf_train': wf_train,
        'workload_encoder': {
            'unique_workloads': unique_workloads,
            'n_features': n_workload_features
        },
        'knob_config_file': args.knob_config,
        'knob_num': args.knob_num,
        'types': types,
        'bounds': bounds.copy(),  # Ensure it's a plain list
        'num_trees': args.num_trees,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    model_path = os.path.join(args.output_dir, 'resource_predictor.joblib')
    joblib.dump(model_data, model_path)
    print(f"  Models saved to {model_path}")
    
    # Save metadata (convert numpy types to native Python types for JSON)
    metadata = {
        'training_time': time.time() - start_time,
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_test': int(len(X_test)),
        'n_features': int(X.shape[1]),
        'n_workload_features': int(n_workload_features),
        'num_trees': int(args.num_trees),
        'test_metrics': {
            'cpu_mape': float(test_metrics['cpu']['mape']),
            'read_io_mape': float(test_metrics['read_io']['mape']),
            'write_io_mape': float(test_metrics['write_io']['mape']),
            'all_pass': bool(all_pass)
        }
    }
    
    import json
    metadata_path = os.path.join(args.output_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"All models <10% MAPE: {'✓ PASS' if all_pass else '✗ FAIL'}")
    print(f"{'='*60}\n")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

