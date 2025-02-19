'''
This script reads your original CSV file, randomly shuffles and splits the data into three subsets (simulating three sites), and writes them into separate CSV files.

python dataset_splitter.py Makhatini2021_data_allfreqs.csv --output_prefix my_output --num_partitions 2

This will split input.csv into 5 parts (my_output_site1.csv, my_output_site2.csv, ..., my_output_site5.csv).
'''

#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse

def split_csv(input_path, output_prefix, num_partitions):
    data = pd.read_csv(input_path)
    np.random.seed(42)
    data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
    n_total = len(data_shuffled)
    
    # Compute split indices
    split_indices = [int(n_total * (i / num_partitions)) for i in range(1, num_partitions)]
    
    # Split data into partitions
    data_splits = np.split(data_shuffled, split_indices)
    
    # Save partitions to separate CSV files
    for i, df in enumerate(data_splits, start=1):
        output_file = f"{output_prefix}_site{i}.csv"
        df.to_csv(output_file, index=False)
        print(f"  {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple random partitions.")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("--output_prefix", type=str, default="site", help="Prefix for output CSV files")
    parser.add_argument("--num_partitions", type=int, default=3, help="Number of partitions to split into")
    
    args = parser.parse_args()
    
    split_csv(args.input_csv, args.output_prefix, args.num_partitions)

