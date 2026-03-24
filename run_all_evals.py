import os
import glob
import subprocess
import tensorflow.compat.v1 as tf
import sys
tf.disable_v2_behavior()

os.environ['TF_USE_LEGACY_KERAS'] = '1'
# Force utf-8 encoding for subprocesses
os.environ['PYTHONIOENCODING'] = 'utf-8'

log_dirs = glob.glob('logs/vrp*')
print(f"Found {len(log_dirs)} log directories.")

for log_dir in log_dirs:
    task = os.path.basename(log_dir).split('-')[0]
    model_dir = os.path.join(log_dir, 'model')
    ckpt = tf.train.latest_checkpoint(model_dir)
    if ckpt:
        print(f"==================================================")
        print(f"Running view_routes for {log_dir} ")
        print(f"  Task: {task}")
        print(f"  Ckpt: {ckpt}")
        print(f"==================================================")
        cmd = [
            sys.executable,
            "view_routes.py",
            "--is_train=False",
            f"--task={task}",
            f"--load_path={ckpt}",
            "--n_show=4"
        ]
        
        try:
            subprocess.run(cmd, check=True, env=os.environ.copy())
            print(f"\n[✓] Successfully finished {log_dir}\n")
        except Exception as e:
            print(f"\n[!] Failed for {log_dir}: {e}\n")
    else:
        print(f"No checkpoint found in {model_dir}\n")
