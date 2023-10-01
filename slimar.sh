#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=def-hasanc
#SBATCH --job-name=slimArtext
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.10 scipy-stack gcc arrow cuda cudnn opencv

source ~/ENV_TORCH/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python main.py 