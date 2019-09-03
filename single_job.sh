#!/bin/bash

#PBS -l select=3:ncpus=10:mem=3GB
#PBS -l walltime=24:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@studenti.unitn.it
#PBS -m be

./priority_sort_task/priority_sort_task.py -lr=$1 -checkpoint_dir=$2 -sequence_max_length=8
