#!/bin/bash

# Possible learning rate
LRATE=("1.000e-04", "7.525e-05", "5.050e-05", "2.575e-05", "1.000e-06")

# Generate a job for each of the learning rates.
for r in ${LRATE[@]}:
  qsub -- single_job.sh $r
