#!/bin/sh
# GE options
#$ -N DCGAN
#$ -l h_rt=48:00:00
#$ -pe gpu-titanx 1
#$ -l h_vmem=32G
#$ -wd /exports/csce/eddie/geos/groups/eddie_geos_eps_eip/Hugo/GAN/PyTorch-StudioGAN
#$ -e /home/s1959730/qoutput/
#$ -o /home/s1959730/qoutput/
#$ -m beas
#$ -M hugo.bloem@ed.ac.uk

# initialise environment modules
. /etc/profile.d/modules.sh

# Load Python
module load cuda/10.2.89
module load anaconda
source activate studioGAN

WANDB_DIR="/exports/eddie/scratch/s1959730/wandb"

python src/main.py \
--project gpm_carb \
--train \
-cfg src/configs/GPM_carb/DCGAN_small.yaml \
-data ./data/gpm_carb_64/ \
-save ./output_carb/ \
--save_fake_images \
--num_workers 0 \
--load_train_hdf5 \
--save_every 1000 \
-e \
-lgv \
-lgv_std 1
