#!/bin/bash
export PATH=$PATH:/opt/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIRBARY_PATH:/opt/openmpi/lib:/usr/local/cuda/include:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

# BUILD EACH PYTHON VERSION - ENSURE DEPENDENCIES ARE INSTALLED

TF_VERSION=1.12

for vpython in python3; do

  echo "Build TensorFlow for Python version:", ${vpython}
  
  # PATH TO git clone tensorflow repo download
  TF_ROOT=/root/tensorflow

  cd $TF_ROOT

  git checkout r${TF_VERSION}

  export PYTHON_BIN_PATH=$(which ${vpython})
  export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
  export PYTHONPATH=${TF_ROOT}/lib
  export PYTHON_ARG=${TF_ROOT}/lib

  export TF_NEED_CUDA=1
  export TF_CUDA_VERSION=10
  export TF_CUDA_COMPUTE_CAPABILITIES="7.0"
  export CUDA_TOOLKIT_PATH=/usr/local/cuda
  export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
  export NCCL_INSTALL_PATH=/usr/lib/x86_64-linux-gnu/nccl
  export TF_CUDA_CLANG=0
  export TF_CUDNN_VERSION=7.4
  export TF_NCCL_VERSION=2.3

  export TF_NEED_AWS=0
  export TF_NEED_GDR=0
  export TF_NEED_GCP=0
  export TF_NEED_S3=0
  export TF_NEED_KAFKA=0
  export TF_NEED_OPENCL_SYCL=0
  export TF_NEED_HDFS=0
  export TF_NEED_OPENCL=0
  export TF_NEED_JEMALLOC=1
  export TF_ENABLE_XLA=0
  export TF_NEED_VERBS=0
  export TF_NEED_ROCM=0

  export TF_NEED_MKL=0
  export TF_DOWNLOAD_MKL=0
  export TF_NEED_MPI=1
  export MPI_PATH=/opt/openmpi
  export GCC_HOST_COMPILER_PATH=$(which gcc)
  export TF_NEED_TENSORRT=0
  export CC_OPT_FLAGS="-march=native"
  export TF_SET_ANDROID_WORKSPACE=0


  # BAZEL BUILD
  bazel clean
  ./configure

  bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

  # build TF PIP PACKAGE
  mkdir -p ${TF_ROOT}/pip/tensorflow_pkg
  bazel-bin/tensorflow/tools/pip_package/build_pip_package ${TF_ROOT}/pip/tensorflow_pkg

  if [ "$vpython" = "python2" ]; then
    pip2 install ${TF_ROOT}/pip/tensorflow_pkg/tensorflow-${TF_VERSION}.0-cp2*
  elif [ "$vpython" = "python3" ]; then
    pip3 install ${TF_ROOT}/pip/tensorflow_pkg/tensorflow-${TF_VERSION}.0-cp3*
  fi

done
