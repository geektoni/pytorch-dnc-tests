#!/bin/bash

#PBS -l select=3:ncpus=10:mem=3GB
#PBS -l walltime=48:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@studenti.unitn.it
#PBS -V
#PBS -m be

~/master/priority_sort_task/priority_sort_task.py -lr=1.000e-04 -mem_size=20 -mem_slot=128 -sequence_max_length=${elements} -non_uniform_priority
