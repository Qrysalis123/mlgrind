import matplotlib.pyplot as plt
import numpy as np
import re
import sys

# Usage: python plot_loss.py log1.txt log2.txt
# Or edit log_files list below

log_dir = "kv_cpu_testing/logs"
log_file = "train.txt"

plt.figure(figsize=(12, 6))

losses = []
with open(log_dir + "/" + log_file) as f:
    for line in f:
        m = re.search(r'loss (\d+\.\d+)', line)
        if m:
            losses.append(float(m.group(1)))

label = log_file.split("/")[-1].replace(".txt", "")
plt.plot(losses, label=f"{label} (raw)")

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
