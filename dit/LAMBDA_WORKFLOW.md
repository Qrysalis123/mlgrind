# Lambda Labs Training Workflow

## 1. SSH into Instance

```bash
ssh ubuntu@<instance-ip>
```

Or with key:
```bash
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>
```

## 2. Sync Files from Local to Remote

From your local laptop, run:

```bash
# sync entire dit folder
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
  ~/kitchen/mlgrind/dit/ ubuntu@<instance-ip>:/home/ubuntu/dit/

# or just specific files
rsync -avz dit/gpu_train.py dit/model.py dit/data.py \
  ubuntu@<instance-ip>:/home/ubuntu/dit/
  
rsync -avz gpu_train.py ubuntu@150.136.43.108:/home/ubuntu/meow/
```

## 3. Setup on Remote (First Time Only)

```bash
# install dependencies
pip install torch tiktoken datasets einops

```

## 4. Run Training

Single GPU (for testing):
```bash
cd /home/ubuntu/dit
python gpu_train.py
```

Multi-GPU with torchrun:
```bash
cd /home/ubuntu/dit
torchrun --standalone --nproc_per_node=8 gpu_train.py
```

Check GPUs config
```bash
nvidia-smi --list-gpus

nvidia-smi

nvidia-smi -L | wc -l

# node
nproc

# live GPU monitoring
watch -n 1 nvidia-smi

# disk space
df -h

# RAM
free -h

# CUDA version
nvcc --version

# Get GPU count, then use it
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) gpu_train.py




```

## 5. Run in Background (Survives SSH Disconnect)

Using tmux:
```bash
# create new session
tmux new -s train

# run training
torchrun --standalone --nproc_per_node=8 gpu_train.py

# detach: Ctrl+B, then D

# reattach later
tmux attach -t train

# kill
tmux kill-session -t train
```

Or using nohup:
```bash
nohup torchrun --standalone --nproc_per_node=8 gpu_train.py > train.log 2>&1 &

# check progress
tail -f train.log
```

## 6. Monitor Training

```bash
# watch GPU usage
watch -n 1 nvidia-smi

# tail the log file
tail -f /mnt/persistent/dit/logs/train_log.txt

# check samples
cat /mnt/persistent/dit/samples/samples_0001000.txt
```

## 7. Resume Training After Crash/Restart

Edit `gpu_train.py`:
```python
init_from = "resume"  # change from "scratch"
```

Then run again:
```bash
torchrun --standalone --nproc_per_node=8 gpu_train.py
```

## 8. Download Checkpoints to Local

From your local laptop:
```bash
# download latest checkpoint
rsync -avz ubuntu@<instance-ip>:/mnt/persistent/dit/checkpoints/ \
  ~/kitchen/mlgrind/dit/checkpoints/

# download samples
rsync -avz ubuntu@<instance-ip>:/mnt/persistent/dit/samples/ \
  ~/kitchen/mlgrind/dit/samples/
```

## 9. Quick Commands Reference

```bash
# sync local -> remote
rsync -avz ~/kitchen/mlgrind/dit/ ubuntu@<IP>:/home/ubuntu/dit/

# run training
torchrun --standalone --nproc_per_node=8 gpu_train.py

# check GPUs
nvidia-smi

# check logs
tail -f /mnt/persistent/dit/logs/train_log.txt

# download checkpoints
rsync -avz ubuntu@<IP>:/mnt/persistent/dit/ ~/kitchen/mlgrind/dit/outputs/
```
