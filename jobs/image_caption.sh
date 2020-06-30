#!/bin/bash
#SBATCH --job-name=CNN_RNN_model.sh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBTACH --ntasks-per-node=16
#SBATCH --time=72:00:00
#SBATCH --output job_cnn_rnn.out
#SBATCH --error job_cnn_rnn.err

source ~/anaconda3/bin/activate ~/anaconda3/envs/NLP

module load cuda

cd /home/apreci2s/Image_captioning

python main.py
