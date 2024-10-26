#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00

## load the necessary modules
module load u18/python/3.11.2
echo 'Loaded python 3.11.2'

## load the lataest CUDA version.
module load u18/cuda/12.1
echo 'Loaded CUDA 12.1'

## load gcc 9.4 for deepspeed
module load u18/gcc/9.4.0
echo 'Loaded gcc 9.4.0'

## remove venv if it exists
# rm -rf env

## create the virtual environment
python3 -m venv env

## Create and acticate venv to run the code in.
source env/bin/activate
echo 'Activated environment'

## Upgrade pip to the latest version.
pip install --upgrade pip

## install the libraries.
pip install -r requirements.txt
# accelerate config --config_file acc_config.yaml

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

## Running the training.
python training.py
# accelerate launch --config_file acc_config.yaml --main_process_port 0 training.py
echo "Training completed"

## Move the saved checkpoints to /share
scp -r /scratch/anlp_peft/  ada:/share1/"$USER"/anlp/
echo "Checkpoints moved to /share"
