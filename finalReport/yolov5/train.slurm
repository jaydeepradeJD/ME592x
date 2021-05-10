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

#python train.py --img-size 416 --batch-size 16 --epoch 200 --data projectdata.yaml --single-cls --weights yolov5s.pt --cfg ./models/yolov5s.yaml --device 0 --name yolov5s_is_416_bs_16_e_200

#python train.py --img-size 416 --batch-size 16 --epoch 200 --data projectdata.yaml --single-cls --weights yolov5m.pt --cfg ./models/yolov5m.yaml --device 0 --name yolov5m_is_416_bs_16_e_200

#python train.py --img-size 416 --batch-size 16 --epoch 200 --data projectdata.yaml --single-cls --weights yolov5l.pt --cfg ./models/yolov5l.yaml --device 0 --name yolov5l_is_416_bs_16_e_200

python train.py --img-size 416 --batch-size 16 --epoch 200 --data projectdata.yaml --single-cls --weights yolov5x.pt --cfg ./models/yolov5x.yaml --device 0 --name yolov5x_is_416_bs_16_e_200
