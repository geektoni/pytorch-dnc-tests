#!/bin/bash

# Possible learning rate
LRATE=("1.000e-04", "7.525e-05", "5.050e-05", "2.575e-05", "1.000e-06")

ELEMS=("8" "4")
PRIOR=("")

# Generate a job for each of the learning rates.
index=0
for e in ${ELEMS[@]}; do
  export elements=${e}
  qsub -V -N "DNC_p_${e}" -q common_cpuQ single_job.sh
  qsub -V -N "DNC_p_${e}_random" -q common_cpuQ single_job_non_uniform.sh
  index=$((index+1))
done
