#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 1 processor core(s) per node 
#SBATCH --mem=20G   # maximum memory per node
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="yolo"
#SBATCH --mail-user=jrrade@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HEREmodule load miniconda3
module load miniconda3
source activate yolov3

#python train.py --img-size 416 --batch-size 16 --epoch 500 --data projectdata.yaml --single-cls --weights yolov3.pt --cfg ./models/yolov3.yaml --device 0 --name is_416_bs_16_e_500

#python train.py --img-size 416 --batch-size 16 --epoch 500 --data projectdata.yaml --single-cls --weights yolov3-tiny.pt --cfg ./models/yolov3-tiny.yaml --device 0 --name yolo_tiny_is_416_bs_16_e_500

python train.py --img-size 416 --batch-size 16 --epoch 500 --data projectdata.yaml --single-cls --weights yolov3-spp.pt --cfg ./models/yolov3-spp.yaml --device 0 --name yolo_spp_is_416_bs_16_e_500
