#!/bin/bash

# Possible learning rate
LRATE=("1.000e-04", "7.525e-05", "5.050e-05", "2.575e-05", "1.000e-06")

# Generate a job for each of the learning rates.
index=0
for r in ${LRATE[@]}:
  TEMP_NAME="./learning_rate_$index"
  qsub -- single_job.sh ${r} ${TEMP_NAME}
  index=$((index+1))
