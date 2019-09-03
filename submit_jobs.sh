#!/bin/bash

# Possible learning rate
LRATE=("1.000e-04", "7.525e-05", "5.050e-05", "2.575e-05", "1.000e-06")

# Generate a job for each of the learning rates.
index=0
for r in ${LRATE[@]}:
  TEMP_NAME="./learning_rate_$index"
  qsub -q common_cpuQ -v learning_rate=${r},check_dir=${TEMP_NAME} singe_job.sh
  index=$((index+1))
