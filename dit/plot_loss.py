import matplotlib.pyplot as plt
import re
import sys

# Usage: python plot_loss.py <log_file>
# Example: python plot_loss.py poem_overfit/logs/train.txt

if len(sys.argv) < 2:
    print("Usage: python plot_loss.py <log_file>")
    sys.exit(1)

log_file = sys.argv[1]


plt.figure(figsize=(12, 6))

losses = []
with open(log_file) as f:
    for line in f:
        m = re.search(r'loss (\d+\.\d+)', line)
        if m:
            losses.append(float(m.group(1)))

label = log_file.split("/")[-1].replace(".txt", "")
plt.plot(losses[240:], label=label)

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
