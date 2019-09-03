#!/bin/bash

#PBS -l select=3:ncpus=12:mem=10GB
#PBS -l walltime=72:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@studenti.unitn.it
#PBS -m be

./priority_sort_task/priority_sort_task.py
