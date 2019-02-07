#!/bin/bash
export PATH=$PATH:/opt/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIRBARY_PATH:/opt/openmpi/lib:/usr/local/cuda/include:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

HOROVOD_GPU_ALLREDUCE=NCCL pip3 install --no-cache-dir horovod

#HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_ALLGATHER=MPI HOROVOD_GPU_BROADCAST=MPI pip install --no-cache-dir horovod
