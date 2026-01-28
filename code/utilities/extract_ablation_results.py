#!/usr/bin/env python3
"""
Extract ablation study results from the markdown file and create an Excel file
"""

import re
import pandas as pd
from collections import defaultdict
file_name = 'ae_kan'
def parse_namespace(namespace_str):
    """Parse namespace string to extract key parameters"""
    # Extract key parameters using regex
    patterns = {
        'lr_rule': r'lr_rule=([0-9.]+)',
        'lr_dkm': r'lr_dkm=([0-9.]+)', 
        'cluster_numbers': r'cluster_numbers=(\d+)',
        'alpha': r'alpha=(\d+)',
        'task': r"task='([^']+)'",
        'data_format': r"data_format='([^']+)'",
        'lambda_dkm': r'lambda_dkm=([0-9.]+)'
    }
    
    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, namespace_str)
        if match:
            if key in ['lr_rule', 'lr_dkm','lambda_dkm']:
                params[key] = float(match.group(1))
            elif key in ['cluster_numbers', 'alpha']:
                params[key] = int(match.group(1))
            else:
                params[key] = match.group(1)
        else:
            params[key] = None
    
    return params

def extract_final_test_accuracy(text_block):
    """Extract the final test accuracy from a text block"""
    # Look for the last occurrence of acc_test in the block
    acc_pattern = r'acc_test:([0-9.]+)'
    matches = re.findall(acc_pattern, text_block)
    
    if matches:
        return float(matches[-1])  # Return the last (final) accuracy
    return None

def main():
    task_name = 'ae_kan'
    # Read the ablation.md file
    with open(f'./gammaILP/cache/{file_name}.md', 'r') as f:
        content = f.read()
    
    # Split content by Namespace occurrences
    namespace_splits = re.split(r'(Namespace\([^)]+\))', content)
    
    results = []
    
    # Process each experiment
    for i in range(1, len(namespace_splits), 2):
        if i + 1 < len(namespace_splits):
            namespace_str = namespace_splits[i]
            experiment_content = namespace_splits[i + 1]
            
            # Parse namespace parameters
            params = parse_namespace(namespace_str)
            
            # Extract final test accuracy
            final_acc = extract_final_test_accuracy(experiment_content)
            
            if final_acc is not None:
                result = {
                    'Task': params.get('task'),
                    'Data_Format': params.get('data_format'),
                    'LR_Rule': params.get('lr_rule'),
                    'LR_DKM': params.get('lr_dkm'),
                    'Cluster_Numbers': params.get('cluster_numbers'),
                    'Alpha': params.get('alpha'),
                    'Lambda_DKM': params.get('lambda_dkm'),
                    'Final_Test_Accuracy': final_acc
                }
                results.append(result)
                
                print(f"Experiment: {params.get('task')} | LR_Rule: {params.get('lr_rule')} | "
                      f"LR_DKM: {params.get('lr_dkm')} | Clusters: {params.get('cluster_numbers')} | "
                      f"Alpha: {params.get('alpha')} | Lambda_DKM: {params.get('lambda_dkm')} | Final Acc: {final_acc}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by task, then by other parameters
    df = df.sort_values(['Task', 'LR_Rule', 'LR_DKM', 'Cluster_Numbers', 'Alpha','Lambda_DKM'])
    
    # Save to Excel
    output_file = f'./gammaILP/cache/{file_name}.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save all results
        df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Create separate sheets for each task
        for task in df['Task'].unique():
            if task:
                task_df = df[df['Task'] == task]
                sheet_name = task.replace('/', '_').replace('\\', '_')[:31]  # Excel sheet name limit
                task_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total experiments processed: {len(results)}")
    print(f"Tasks found: {df['Task'].unique()}")
    
    # Print summary statistics
    print("\nSummary by Task:")
    for task in df['Task'].unique():
        if task:
            task_df = df[df['Task'] == task]
            print(f"\n{task}:")
            print(f"  Number of experiments: {len(task_df)}")
            print(f"  Best accuracy: {task_df['Final_Test_Accuracy'].max():.4f}")
            print(f"  Worst accuracy: {task_df['Final_Test_Accuracy'].min():.4f}")
            print(f"  Average accuracy: {task_df['Final_Test_Accuracy'].mean():.4f}")

if __name__ == "__main__":
    main()
