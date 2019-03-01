## Template for Creating Multi-node Parallel Batch Docker Images

Template scripts to setup Docker Images compatible with running on MNP Batch

### License Summary

This sample code is made available under a modified MIT license. See the LICENSE file.

### Tensorflow Deployment
To build a Tensorflow reference docker image compatible with running tightly coupled multi-node parallel batch jobs on AWS Batch. Build platform requires installation of nvidia-docker2.

```bash
git clone https://github.com/aws-samples/aws-mnpbatch-template.git
cd aws-mnpbatch-template
docker build -t nvidia/mnp-batch-tensorflow .
```

### Custom Application Deployment
The Dockerfile can mostly be reused for your application, It installs the following stack:
```
Ubuntu 18.04 nvidia/cuda base docker image
APT packages for dependencies
SSH SETUP
S3 OPTIMIZATION
CUDA-AWARE MPI 4.0.0
TENSORFLOW INSTALL
IMAGENET DATASET
SUPERVISOR DOCKER CONTAINER STARTUP
```
Thus if you want apply your own customizations and application, you just need to modify the MPI, Tensorflow layers. Also custom build scripts are located in ```conf/```.

Finally replace the section in ```supervised-scripts/mpi-run.sh``` to support the MPI startup of your custom application. The script logic will prepare the mpi machine file. If your node contains GPUs then the ```slots=```  with be the number of GPUs per node. 
```bash
if [ -x "$(command -v nvidia-smi)" ] ; then
      NUM_GPUS=$(ls -l /dev/nvidia[0-9] | wc -l)
      availablecores=$NUM_GPUS
  else
      availablecores=$(nproc)
  fi
```
If not then it will based on the vCPUs/Cores where applicable and passed as ```${HOST_FILE_PATH}-deduped```. Any extra MPI parameters at job runtime and will be passed into the ```$EXTRA_MPI_PARAMS```.
```bash
wait_for_nodes () {
	.
	.
	.
  aws s3 cp $S3_INPUT $SCRATCH_DIR
  #tar -xvf $SCRATCH_DIR/*.tar.gz -C $SCRATCH_DIR

  cd $SCRATCH_DIR
  export INTERFACE=eth0
  export MODEL_HOME=/root/deep-learning-models/models/resnet/tensorflow
  /opt/openmpi/bin/mpirun --allow-run-as-root -np $MPI_GPUS --machinefile ${HOST_FILE_PATH}-deduped -mca plm_rsh_no_tree_spawn 1 \
                          -bind-to socket -map-by slot \
                          -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
                          -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
                          -x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_include $INTERFACE \
                          $EXTRA_MPI_PARAMS -x TF_CPP_MIN_LOG_LEVEL=0 \
                          python3 -W ignore $MODEL_HOME/train_imagenet_resnet_hvd.py \
                          --data_dir $JOB_DIR --num_epochs 90 -b $BATCH_SIZE \
                          --lr_decay_mode poly --warmup_epochs 10 --clear_log
  sleep 2

  #tar -czvf $JOB_DIR/batch_output_$AWS_BATCH_JOB_ID.tar.gz $SCRATCH_DIR/*
  #aws s3 cp $JOB_DIR/batch_output_$AWS_BATCH_JOB_ID.tar.gz $S3_OUTPUT
 }
```
Once built you can commit this docker image to your AWS Elastic Container Registry (ECR) using [these instructions](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html). 

