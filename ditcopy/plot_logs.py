#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob

# Get log file from argument or use latest
if len(sys.argv) > 1:
    log_file = sys.argv[1]
else:
    log_files = sorted(glob.glob('logs/*.log'))
    if not log_files:
        print("No log files found")
        sys.exit(1)
    log_file = log_files[-1]

print(f"Plotting {log_file}")

# Read and plot
df = pd.read_csv(log_file, comment='#')
plt.plot(df['step'], df['loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title(f'Training Loss')
plt.grid(True, alpha=0.3)
plt.show()
