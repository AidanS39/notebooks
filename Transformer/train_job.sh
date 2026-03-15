#!/bin/bash

#SBATCH --chdir=./                       # Set the working directory
#SBATCH --mail-user=stonera3@tcnj.edu    # Who to send emails to
#SBATCH --mail-type=ALL                  # Send emails on start, end and failure
#SBATCH --job-name=train_transformer     # Name to show in the job queue
#SBATCH --output=./output/job.%j.out     # Name of stdout output file (%j expands to jobId)
#SBATCH --ntasks=16                      # Total number of mpi tasks requested (CPU Cores basically)
#SBATCH --nodes=1                        # Total number of nodes requested
#SBATCH --partition=gpu			         # Partition (a.k.a. queue) to use
#SBATCH --gres=gpu:1			         # Select GPU resource (# after : indicates how many)
#SBATCH --constraint=l40s                # use rtxa5000, l40s or gtx1080ti here to limit GPU selection
#SBATCH --time=5-00:00:00                # Max run time (days-hh:mm:ss) ... adjust as necessary

# set model parameters
D_MODEL=1024
N_HEADS=8
N_LAYERS=8
D_UP=2048
N_EPOCHS=5

while getopts "m:h:l:u:e:" flag; do
    case "${flag}" in
        m) 
            D_MODEL=$OPTARG ;;
        h) 
            N_HEADS=$OPTARG ;;
        l) 
            N_LAYERS=$OPTARG ;;
        u)
            D_UP=$OPTARG ;;
        e)
            N_EPOCHS=$OPTARG ;;
    esac
done

# training script
python3 -u train.py -m $D_MODEL -h $N_HEADS -l $N_LAYERS -u $D_UP -e $N_EPOCHS