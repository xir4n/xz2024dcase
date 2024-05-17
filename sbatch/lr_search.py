import os

# Define constants.
script_name = "run_training"
script_path = os.path.abspath(os.path.join("..", script_name)) + ".py"
batch_size = 512
project_name = os.path.basename(__file__)[:-3]
subset = 5

lrs = [0.1, 0.08, 0.05, 0.03, 0.01, 0.001]

# Create folder.
sbatch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), project_name)
os.makedirs(sbatch_dir, exist_ok=True)

for lr in lrs:
    experiment_name = f"lr_{lr}"
    file_name = experiment_name + ".sbatch"
    file_path = os.path.join(sbatch_dir, file_name)
    sav_dir = os.path.join("./log", project_name)
    
    # Generate file.
    with open(file_path, "w") as f:
        cmd_args = [
            script_path, 
            f"--project_name {project_name}",
            f"--experiment_name {experiment_name}",
            f"--subset {subset}",
            f"--alpha {1}",
            f"--beta {1}",
            f"--mixstyle_p {0}",
            f"--dir_prob {0}",
            f"--sav_dir {sav_dir}",
            f"--batch_size {batch_size}",
            f"--lr {lr}",
            f"--warmup_steps {0}",
        ]
        f.write("#!/bin/bash\n")
        f.write("\n")
        f.write("#BATCH --job-name=" + experiment_name + "\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH -C v100-32g\n")
        f.write("#SBATCH --tasks-per-node=1\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --cpus-per-task=10\n")
        f.write("#SBATCH --hint=nomultithread\n")

        f.write("#SBATCH --time=20:00:00\n")
        f.write("#SBATCH --account=nvz@v100\n")
        f.write("#SBATCH --output=" + experiment_name + "_%j.out\n")
        f.write("\n")
        f.write("module purge\n")
        f.write("\n")
        f.write("module load anaconda-py3/2023.09\n")
        f.write("\n")
        f.write("export PATH=$WORK/.local/bin:$PATH\n")
        f.write("conda activate dcase\n")
        f.write(" ".join(["python"] + cmd_args) + "\n")
        f.write("\n")

# Open shell file.
file_path = os.path.join(sbatch_dir, script_name.split("_")[0] + ".sh") #./structure_alpha_beta/run.sh
with open(file_path, "w") as f:
    # Print header.
    f.write(
        "# Find "
    )
    f.write("\n")
    for lr in lrs:
        experiment_name = f"lr_{lr}"
        file_name = experiment_name + ".sbatch"
        file_path = os.path.join(sbatch_dir, file_name)
        sbatch_str = "sbatch " + file_path
        # Write SBATCH command to shell file.
        f.write(sbatch_str + "\n")

# Grant permission to execute the shell file.
# https://stackoverflow.com/a/30463972
mode = os.stat(file_path).st_mode
mode |= (mode & 0o444) >> 2
os.chmod(file_path, mode)