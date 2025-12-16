#!/usr/bin/env python3
"""
Compare OpAdviser results against ground truth baseline.

Usage:
    python scripts/compare_results.py --opadviser=sbrw_opadviser --ground_truth=sbrw_ground_truth
    python scripts/compare_results.py --opadviser=history_sbrw_opadviser.json --ground_truth=history_sbrw_ground_truth.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np


def load_history(filepath: str) -> List[Dict]:
    """Load tuning history from JSON file.
    
    Args:
        filepath: Path to history JSON file or task_id.
        
    Returns:
        List of observation dictionaries.
    """
    # Handle both full path and task_id
    if not filepath.endswith('.json'):
        filepath = f"repo/history_{filepath}.json"
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get('data', [])


def get_best_config(history: List[Dict], metric: str = 'tps') -> Tuple[Dict, float, int]:
    """Find the best configuration from history.
    
    Args:
        history: List of observation dictionaries.
        metric: Metric to optimize (default: 'tps').
        
    Returns:
        Tuple of (best_config, best_metric_value, iteration_index).
    """
    best_idx = 0
    best_value = 0
    for i, obs in enumerate(history):
        value = obs.get('external_metrics', {}).get(metric, 0)
        if value > best_value:
            best_value = value
            best_idx = i
    best_obs = history[best_idx]
    best_config = best_obs.get('configuration', {})
    return best_config, best_value, best_idx + 1  # 1-indexed


def get_worst_config(history: List[Dict], metric: str = 'tps', 
                     exclude_failures: bool = False) -> Tuple[Dict, float, int]:
    """Find the worst configuration from history.
    
    Args:
        history: List of observation dictionaries.
        metric: Metric to find worst for (default: 'tps').
        exclude_failures: If True, exclude runs with TPS < 10% of median (likely failures).
        
    Returns:
        Tuple of (worst_config, worst_metric_value, iteration_index).
    """
    # Filter out completely failed runs (TPS = 0 or very close to 0)
    valid_history = [(i, obs) for i, obs in enumerate(history) 
                     if obs.get('external_metrics', {}).get(metric, 0) > 0.1]
    
    if not valid_history:
        # If all runs failed, return the first one
        return history[0].get('configuration', {}), 0, 1
    
    if exclude_failures:
        # Calculate median TPS
        tps_values = [obs.get('external_metrics', {}).get(metric, 0) for _, obs in valid_history]
        median_tps = sorted(tps_values)[len(tps_values) // 2]
        # Exclude runs with TPS < 10% of median (likely failures/outliers)
        threshold = median_tps * 0.1
        valid_history = [(i, obs) for i, obs in valid_history 
                         if obs.get('external_metrics', {}).get(metric, 0) >= threshold]
    
    if not valid_history:
        return history[0].get('configuration', {}), 0, 1
    
    worst_idx, worst_obs = min(valid_history, 
                                key=lambda x: x[1].get('external_metrics', {}).get(metric, float('inf')))
    worst_value = worst_obs.get('external_metrics', {}).get(metric, 0)
    worst_config = worst_obs.get('configuration', {})
    return worst_config, worst_value, worst_idx + 1  # 1-indexed


def get_default_config_performance(history: List[Dict], metric: str = 'tps') -> Tuple[Dict, float]:
    """Get the performance of the default/initial configuration (iteration 1).
    
    Args:
        history: List of observation dictionaries.
        metric: Metric to check (default: 'tps').
        
    Returns:
        Tuple of (default_config, default_metric_value).
    """
    if not history:
        return {}, 0
    
    default_obs = history[0]  # First iteration uses default config
    default_value = default_obs.get('external_metrics', {}).get(metric, 0)
    default_config = default_obs.get('configuration', {})
    return default_config, default_value


def get_convergence_iteration(history: List[Dict], target_ratio: float = 0.95, 
                               metric: str = 'tps') -> Optional[int]:
    """Find iteration where performance reaches target_ratio of best.
    
    Args:
        history: List of observation dictionaries.
        target_ratio: Target ratio of best performance (default: 0.95 = 95%).
        metric: Metric to optimize.
        
    Returns:
        Iteration number where target was reached, or None if never reached.
    """
    _, best_value, _ = get_best_config(history, metric)
    target_value = best_value * target_ratio
    
    running_best = 0
    for i, obs in enumerate(history):
        value = obs.get('external_metrics', {}).get(metric, 0)
        running_best = max(running_best, value)
        if running_best >= target_value:
            return i + 1  # 1-indexed
    
    return None


def get_convergence_curve(history: List[Dict], metric: str = 'tps') -> List[float]:
    """Get the best-so-far convergence curve.
    
    Args:
        history: List of observation dictionaries.
        metric: Metric to track.
        
    Returns:
        List of best-so-far values at each iteration.
    """
    curve = []
    running_best = 0
    for obs in history:
        value = obs.get('external_metrics', {}).get(metric, 0)
        running_best = max(running_best, value)
        curve.append(running_best)
    return curve


def print_comparison_report(op_history: List[Dict], gt_history: List[Dict], 
                            metric: str = 'tps') -> None:
    """Print a detailed comparison report.
    
    Args:
        op_history: OpAdviser tuning history.
        gt_history: Ground truth tuning history.
        metric: Metric to compare.
    """
    # Get best results
    op_config, op_best, op_best_iter = get_best_config(op_history, metric)
    gt_config, gt_best, gt_best_iter = get_best_config(gt_history, metric)
    
    # Get worst results (for performance improvement calculation)
    op_worst_config, op_worst, op_worst_iter = get_worst_config(op_history, metric)
    gt_worst_config, gt_worst, gt_worst_iter = get_worst_config(gt_history, metric)
    
    # Get "practical worst" (excluding likely failed configs)
    _, op_practical_worst, op_practical_worst_iter = get_worst_config(op_history, metric, exclude_failures=True)
    _, gt_practical_worst, gt_practical_worst_iter = get_worst_config(gt_history, metric, exclude_failures=True)
    
    # Get default configuration performance
    gt_default_config, gt_default = get_default_config_performance(gt_history, metric)
    op_default_config, op_default = get_default_config_performance(op_history, metric)
    
    # Combined history for overall worst
    combined_history = op_history + gt_history
    _, combined_worst, _ = get_worst_config(combined_history, metric)
    _, combined_practical_worst, _ = get_worst_config(combined_history, metric, exclude_failures=True)
    _, combined_best, _ = get_best_config(combined_history, metric)
    
    # Get convergence info
    op_conv_95 = get_convergence_iteration(op_history, 0.95, metric)
    op_conv_90 = get_convergence_iteration(op_history, 0.90, metric)
    gt_conv_95 = get_convergence_iteration(gt_history, 0.95, metric)
    gt_conv_90 = get_convergence_iteration(gt_history, 0.90, metric)
    
    # Calculate efficiency metrics
    ratio = op_best / gt_best if gt_best > 0 else 0
    iterations_saved = len(gt_history) - len(op_history)
    
    # Calculate performance improvement ratios
    op_improvement = op_best / op_worst if op_worst > 0 else float('inf')
    gt_improvement = gt_best / gt_worst if gt_worst > 0 else float('inf')
    overall_improvement = combined_best / combined_worst if combined_worst > 0 else float('inf')
    
    # Practical improvement (excluding likely failed configs)
    op_practical_improvement = op_best / op_practical_worst if op_practical_worst > 0 else float('inf')
    gt_practical_improvement = gt_best / gt_practical_worst if gt_practical_worst > 0 else float('inf')
    overall_practical_improvement = combined_best / combined_practical_worst if combined_practical_worst > 0 else float('inf')
    
    print("\n" + "=" * 70)
    print("                    OPADVISER EVALUATION REPORT")
    print("=" * 70)
    
    print("\n📊 SUMMARY")
    print("-" * 70)
    print(f"  OpAdviser iterations:     {len(op_history)}")
    print(f"  Ground Truth iterations:  {len(gt_history)}")
    print(f"  Iterations saved:         {iterations_saved} ({iterations_saved/len(gt_history)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("              🚀 PERFORMANCE IMPROVEMENT (TUNING IMPACT)")
    print("=" * 70)
    
    print("\n🔧 DEFAULT CONFIGURATION (MySQL Defaults)")
    print("-" * 70)
    print(f"  Ground Truth default {metric.upper()}: {gt_default:,.2f} (iteration 1)")
    print(f"  OpAdviser default {metric.upper()}:    {op_default:,.2f} (iteration 1)")
    
    print("\n📉 WORST CONFIGURATION (Bad Tuning)")
    print("-" * 70)
    print(f"  Ground Truth worst {metric.upper()}:  {gt_worst:,.2f} (iteration {gt_worst_iter})")
    print(f"  OpAdviser worst {metric.upper()}:     {op_worst:,.2f} (iteration {op_worst_iter})")
    print(f"  Overall worst {metric.upper()}:       {combined_worst:,.2f}")
    
    print("\n📉 PRACTICAL WORST (Excluding Failed Runs)")
    print("-" * 70)
    print(f"  Ground Truth practical worst: {gt_practical_worst:,.2f} (iteration {gt_practical_worst_iter})")
    print(f"  OpAdviser practical worst:    {op_practical_worst:,.2f} (iteration {op_practical_worst_iter})")
    print(f"  Overall practical worst:      {combined_practical_worst:,.2f}")
    
    print("\n📈 BEST CONFIGURATION (After Tuning)")
    print("-" * 70)
    print(f"  Ground Truth best {metric.upper()}:   {gt_best:,.2f} (iteration {gt_best_iter})")
    print(f"  OpAdviser best {metric.upper()}:      {op_best:,.2f} (iteration {op_best_iter})")
    print(f"  Overall best {metric.upper()}:        {combined_best:,.2f}")
    
    print("\n⚡ PERFORMANCE IMPROVEMENT RATIO (best / worst)")
    print("-" * 70)
    print(f"  Ground Truth improvement: {gt_improvement:,.2f}x (best/worst)")
    print(f"  OpAdviser improvement:    {op_improvement:,.2f}x (best/worst)")
    print(f"  Overall improvement:      {overall_improvement:,.2f}x (best/worst)")
    
    print("\n⚡ PRACTICAL IMPROVEMENT (excluding outliers/failures)")
    print("-" * 70)
    print(f"  Ground Truth practical:   {gt_practical_improvement:,.2f}x")
    print(f"  OpAdviser practical:      {op_practical_improvement:,.2f}x")
    print(f"  Overall practical:        {overall_practical_improvement:,.2f}x")
    
    # Improvement vs default (more meaningful for real-world)
    gt_vs_default = gt_best / gt_default if gt_default > 0 else float('inf')
    op_vs_default = op_best / op_default if op_default > 0 else float('inf')
    best_vs_default = combined_best / max(gt_default, op_default) if max(gt_default, op_default) > 0 else float('inf')
    
    print("\n⚡ IMPROVEMENT VS DEFAULT CONFIGURATION")
    print("-" * 70)
    print(f"  Ground Truth vs default:  {gt_vs_default:,.2f}x")
    print(f"  OpAdviser vs default:     {op_vs_default:,.2f}x")
    print(f"  Best overall vs default:  {best_vs_default:,.2f}x")
    
    # Highlight if we achieved the target improvement
    TARGET_IMPROVEMENT = 4.5
    max_improvement = max(overall_improvement, overall_practical_improvement, best_vs_default)
    
    print(f"\n  🎯 Target improvement:    {TARGET_IMPROVEMENT}x")
    print("-" * 70)
    if overall_improvement >= TARGET_IMPROVEMENT:
        print(f"  ✅ SUCCESS: Achieved {overall_improvement:.2f}x improvement (best/worst)")
    if overall_practical_improvement >= TARGET_IMPROVEMENT:
        print(f"  ✅ SUCCESS: Achieved {overall_practical_improvement:.2f}x practical improvement")
    if best_vs_default >= TARGET_IMPROVEMENT:
        print(f"  ✅ SUCCESS: Achieved {best_vs_default:.2f}x improvement vs default")
    
    if max_improvement < TARGET_IMPROVEMENT:
        print(f"  ⚠️  Below target: best improvement is {max_improvement:.2f}x < {TARGET_IMPROVEMENT}x")
    else:
        print(f"\n  🏆 MAXIMUM IMPROVEMENT ACHIEVED: {max_improvement:.2f}x")
    
    print("\n" + "=" * 70)
    print("              🏆 OPADVISER VS GROUND TRUTH COMPARISON")
    print("=" * 70)
    
    print("\n🏆 BEST PERFORMANCE COMPARISON")
    print("-" * 70)
    print(f"  OpAdviser best {metric.upper()}:      {op_best:,.2f}")
    print(f"  Ground Truth best {metric.upper()}:  {gt_best:,.2f}")
    print(f"  OpAdviser achieves:       {ratio*100:.1f}% of ground truth")
    
    print("\n⏱️  CONVERGENCE SPEED")
    print("-" * 70)
    print(f"  OpAdviser to 90% of its best:    {op_conv_90 or 'N/A'} iterations")
    print(f"  OpAdviser to 95% of its best:    {op_conv_95 or 'N/A'} iterations")
    print(f"  Ground Truth to 90% of its best: {gt_conv_90 or 'N/A'} iterations")
    print(f"  Ground Truth to 95% of its best: {gt_conv_95 or 'N/A'} iterations")
    
    # Check if OpAdviser reached ground truth level
    op_at_gt_90 = get_convergence_iteration(
        op_history, 0.90 * gt_best / (op_best if op_best > 0 else 1), metric
    )
    print(f"\n  OpAdviser to 90% of GT best:     {op_at_gt_90 or 'Not reached'} iterations")
    
    print("\n📈 EVALUATION VERDICT")
    print("-" * 70)
    if ratio >= 0.95:
        print("  ✅ EXCELLENT: OpAdviser achieved ≥95% of ground truth performance")
    elif ratio >= 0.90:
        print("  ✅ GOOD: OpAdviser achieved ≥90% of ground truth performance")
    elif ratio >= 0.80:
        print("  ⚠️  ACCEPTABLE: OpAdviser achieved ≥80% of ground truth performance")
    else:
        print("  ❌ NEEDS IMPROVEMENT: OpAdviser achieved <80% of ground truth")
    
    if iterations_saved > 0 and ratio >= 0.90:
        efficiency = (ratio * 100) / (len(op_history) / len(gt_history) * 100)
        print(f"  📊 Efficiency score: {efficiency:.2f}x (higher is better)")
    
    # Print configuration differences for best configurations
    print("\n" + "=" * 70)
    print("              🔧 BEST CONFIGURATION DETAILS")
    print("=" * 70)
    
    print("\n🏆 Best Ground Truth Configuration:")
    print("-" * 70)
    for knob, value in sorted(gt_config.items()):
        print(f"  {knob}: {value}")
    
    print("\n🏆 Best OpAdviser Configuration:")
    print("-" * 70)
    for knob, value in sorted(op_config.items()):
        print(f"  {knob}: {value}")
    
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare OpAdviser results against ground truth baseline.'
    )
    parser.add_argument(
        '--opadviser', '-o', 
        required=True,
        help='OpAdviser history file or task_id (e.g., sbrw_opadviser)'
    )
    parser.add_argument(
        '--ground_truth', '-g', 
        required=True,
        help='Ground truth history file or task_id (e.g., sbrw_ground_truth)'
    )
    parser.add_argument(
        '--metric', '-m', 
        default='tps',
        help='Metric to compare (default: tps)'
    )
    parser.add_argument(
        '--plot', '-p', 
        action='store_true',
        help='Generate convergence plot'
    )
    
    args = parser.parse_args()
    
    # Load histories
    print(f"\nLoading OpAdviser results: {args.opadviser}")
    op_history = load_history(args.opadviser)
    
    print(f"Loading Ground Truth results: {args.ground_truth}")
    gt_history = load_history(args.ground_truth)
    
    # Print comparison report
    print_comparison_report(op_history, gt_history, args.metric)
    
    # Generate plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            op_curve = get_convergence_curve(op_history, args.metric)
            gt_curve = get_convergence_curve(gt_history, args.metric)
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(op_curve) + 1), op_curve, 
                    label='OpAdviser', linewidth=2, color='blue')
            plt.plot(range(1, len(gt_curve) + 1), gt_curve, 
                    label='Ground Truth', linewidth=2, color='orange', linestyle='--')
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel(f'Best {args.metric.upper()} So Far', fontsize=12)
            plt.title('OpAdviser vs Ground Truth Convergence', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_file = f'comparison_{args.opadviser}_{args.ground_truth}.png'
            plt.savefig(output_file, dpi=150)
            print(f"Convergence plot saved to: {output_file}")
            
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")


if __name__ == '__main__':
    main()

