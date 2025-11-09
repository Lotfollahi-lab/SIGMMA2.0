#!/usr/bin/env python3
"""
Quick analysis script to track BLEEP hyperparameter sweep progress.
Monitors key metrics from training output files.

Usage:
    python analyze_sweep_results.py
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

def parse_metrics_from_log(log_file):
    """Extract metrics from training log file."""
    metrics = defaultdict(list)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    epoch = 0
    for i, line in enumerate(lines):
        # Track epochs
        if 'Checkpoint saved' in line:
            epoch += 1
        
        # Parse alignment metrics
        if '[Alignment]' in line:
            match = re.search(r'pos_gap=([-\d.]+)', line)
            if match:
                metrics['pos_gap'].append(float(match.group(1)))
            
            match = re.search(r'diag=([-\d.]+)', line)
            if match:
                metrics['diag'].append(float(match.group(1)))
                
            match = re.search(r'off_diag=([-\d.]+)', line)
            if match:
                metrics['off_diag'].append(float(match.group(1)))
                
            match = re.search(r'top1=([\d.]+)%', line)
            if match:
                metrics['top1_acc'].append(float(match.group(1)))
        
        # Parse image similarity metrics
        if '[Img_Sim]' in line:
            match = re.search(r'self=([-\d.]+)', line)
            if match:
                metrics['img_self_sim'].append(float(match.group(1)))
            
            match = re.search(r'off_diag=([-\d.]+)', line)
            if match:
                metrics['img_off_diag_sim'].append(float(match.group(1)))
        
        # Parse spot similarity metrics
        if '[Spot_Sim]' in line:
            match = re.search(r'self=([-\d.]+)', line)
            if match:
                metrics['spot_self_sim'].append(float(match.group(1)))
            
            match = re.search(r'off_diag=([-\d.]+)', line)
            if match:
                metrics['spot_off_diag_sim'].append(float(match.group(1)))
        
        # Parse loss
        if 'Loss tensor' in line:
            match = re.search(r'tensor\(([\d.]+)', line)
            if match:
                metrics['loss'].append(float(match.group(1)))
    
    return metrics, epoch

def extract_config_from_log(log_file):
    """Extract hyperparameter config from log filename or content."""
    filename = Path(log_file).name
    
    config = {
        'job_id': 'unknown',
        'temperature': None,
        'learning_rate': None,
        'batch_size': None,
    }
    
    # Try to extract from filename pattern: sweep_JOBINDEX_JOBID.out
    match = re.search(r'sweep_(\d+)_(\d+)\.out', filename)
    if match:
        config['job_id'] = match.group(2)
    
    # Parse hyperparameters from log content
    with open(log_file, 'r') as f:
        content = f.read()
        
        # This would require reading the config from wandb or output
        # For now, we'll infer from output directory if available
    
    return config

def analyze_sweep_results(runs_dir='runs_bleep'):
    """Analyze all sweep results and create summary."""
    
    sweep_files = list(Path(runs_dir).glob('sweep_*.out'))
    
    if not sweep_files:
        print(f"No sweep output files found in {runs_dir}")
        return
    
    print(f"Found {len(sweep_files)} sweep runs\n")
    print("="*80)
    
    results = []
    
    for log_file in sorted(sweep_files):
        print(f"\nAnalyzing: {log_file.name}")
        
        config = extract_config_from_log(log_file)
        metrics, epochs_completed = parse_metrics_from_log(log_file)
        
        if not metrics['pos_gap']:
            print(f"  ⚠️  No metrics found - training may have failed")
            continue
        
        # Calculate summary statistics
        summary = {
            'file': log_file.name,
            'job_id': config['job_id'],
            'epochs': epochs_completed,
            'final_pos_gap': metrics['pos_gap'][-1] if metrics['pos_gap'] else None,
            'max_pos_gap': max(metrics['pos_gap']) if metrics['pos_gap'] else None,
            'final_top1_acc': metrics['top1_acc'][-1] if metrics['top1_acc'] else None,
            'max_top1_acc': max(metrics['top1_acc']) if metrics['top1_acc'] else None,
            'final_loss': metrics['loss'][-1] if metrics['loss'] else None,
            'min_loss': min(metrics['loss']) if metrics['loss'] else None,
            'avg_img_self_sim': sum(metrics['img_self_sim']) / len(metrics['img_self_sim']) if metrics['img_self_sim'] else None,
            'avg_spot_self_sim': sum(metrics['spot_self_sim']) / len(metrics['spot_self_sim']) if metrics['spot_self_sim'] else None,
        }
        
        results.append(summary)
        
        # Print quick summary
        print(f"  Epochs: {epochs_completed}")
        if summary['final_pos_gap'] is not None:
            trend = "📈" if summary['max_pos_gap'] > 0 else "📉"
            print(f"  {trend} pos_gap: {summary['final_pos_gap']:.4f} (max: {summary['max_pos_gap']:.4f})")
        if summary['final_top1_acc'] is not None:
            print(f"  🎯 top1_acc: {summary['final_top1_acc']:.1f}% (max: {summary['max_top1_acc']:.1f}%)")
        if summary['final_loss'] is not None:
            print(f"  📊 loss: {summary['final_loss']:.4f} (min: {summary['min_loss']:.4f})")
    
    print("\n" + "="*80)
    print("\n📊 SUMMARY TABLE\n")
    
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('max_pos_gap', ascending=False)
        print(df[['file', 'epochs', 'max_pos_gap', 'max_top1_acc', 'min_loss']].to_string(index=False))
        
        print("\n🏆 BEST RUNS:")
        print(f"  Best pos_gap: {df.iloc[0]['file']} (pos_gap={df.iloc[0]['max_pos_gap']:.4f})")
        print(f"  Best accuracy: {df.loc[df['max_top1_acc'].idxmax()]['file']} (acc={df['max_top1_acc'].max():.1f}%)")
        print(f"  Best loss: {df.loc[df['min_loss'].idxmin()]['file']} (loss={df['min_loss'].min():.4f})")
    
    return df

if __name__ == '__main__':
    import sys
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else 'runs_bleep'
    
    print("BLEEP Hyperparameter Sweep Analysis")
    print("="*80)
    
    df = analyze_sweep_results(runs_dir)
    
    if df is not None and not df.empty:
        # Save to CSV
        output_file = Path(runs_dir) / 'sweep_summary.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
