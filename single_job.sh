#!/bin/bash

#PBS -l select=3:ncpus=10:mem=3GB
#PBS -l walltime=24:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@studenti.unitn.it
#PBS -m be

~/master/priority_sort_task/priority_sort_task.py -lr=$learning_rate -checkpoint_dir=$check_dir -sequence_max_length=8
