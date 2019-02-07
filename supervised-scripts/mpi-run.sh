#!/bin/bash
echo "################################"
env
echo "################################"

PATH="$PATH:/opt/openmpi/bin/"
BASENAME="${0##*/}"
log () {
  echo "${BASENAME} - ${1}"
}
HOST_FILE_PATH="/tmp/hostfile"
AWS_BATCH_EXIT_CODE_FILE="/tmp/batch-exit-code"


usage () {
  if [ "${#@}" -ne 0 ]; then
    log "* ${*}"
    log
  fi
  cat <<ENDUSAGE
Usage:
export AWS_BATCH_JOB_NODE_INDEX=0
export AWS_BATCH_JOB_NUM_NODES=10
export AWS_BATCH_JOB_MAIN_NODE_INDEX=0
export AWS_BATCH_JOB_ID=string
./mpi-run.sh
ENDUSAGE

  error_exit
}

# Standard function to print an error and exit with a failing return code
error_exit () {
  log "${BASENAME} - ${1}" >&2
  log "${2:-1}" > $AWS_BATCH_EXIT_CODE_FILE
  kill  $(cat /tmp/supervisord.pid)
}

# Check what environment variables are set
if [ -z "${AWS_BATCH_JOB_NODE_INDEX}" ]; then
  usage "AWS_BATCH_JOB_NODE_INDEX not set, unable to determine rank"
fi

if [ -z "${AWS_BATCH_JOB_NUM_NODES}" ]; then
  usage "AWS_BATCH_JOB_NUM_NODES not set. Don't know how many nodes in this job."
fi

if [ -z "${AWS_BATCH_JOB_MAIN_NODE_INDEX}" ]; then
  usage "AWS_BATCH_MULTI_MAIN_NODE_RANK must be set to determine the master node rank"
fi

NODE_TYPE="child"
if [ "${AWS_BATCH_JOB_MAIN_NODE_INDEX}" == "${AWS_BATCH_JOB_NODE_INDEX}" ]; then
  log "Running synchronize as the main node"
  NODE_TYPE="main"
fi

# Check that necessary programs are available
which aws >/dev/null 2>&1 || error_exit "Unable to find AWS CLI executable."


# wait for all nodes to report
wait_for_nodes () {
  log "Running as master node"

  touch $HOST_FILE_PATH
  ip=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
  if [ -x "$(command -v nvidia-smi)" ] ; then
      NUM_GPUS=$(ls -l /dev/nvidia[0-9] | wc -l)
      availablecores=$NUM_GPUS
  else
      availablecores=$(nproc)
  fi
  log "master details -> $ip:$availablecores"
  echo "$ip slots=$availablecores" >> $HOST_FILE_PATH

  lines=$(uniq $HOST_FILE_PATH|wc -l)
  while [ "$AWS_BATCH_JOB_NUM_NODES" -gt "$lines" ]
  do
    log "$lines out of $AWS_BATCH_JOB_NUM_NODES nodes joined, will check again in 1 second"
    sleep 1
    lines=$(uniq $HOST_FILE_PATH|wc -l)
  done
  # Make the temporary file executable and run it with any given arguments
  log "All nodes successfully joined"
  # remove duplicates if there are any.
  awk '!a[$0]++' $HOST_FILE_PATH > ${HOST_FILE_PATH}-deduped
  cat $HOST_FILE_PATH-deduped
  log "executing main MPIRUN workflow"

  #aws s3 cp $S3_INPUT $SCRATCH_DIR
  #tar -xvf $SCRATCH_DIR/*.tar.gz -C $SCRATCH_DIR

  cd $SCRATCH_DIR
  export INTERFACE=eth0
  export MODEL_HOME=/root/deep-learning-models/models/resnet/tensorflow
  /opt/openmpi/bin/mpirun --allow-run-as-root -np $MPI_GPUS --machinefile ${HOST_FILE_PATH}-deduped -mca plm_rsh_no_tree_spawn 1 \
                          -bind-to socket -map-by slot \
                          -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 \
                          -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
                          -x NCCL_SOCKET_IFNAME=$INTERFACE -mca btl_tcp_if_include $INTERFACE \
                          -x TF_CPP_MIN_LOG_LEVEL=0 \
                          python3 -W ignore $MODEL_HOME/train_imagenet_resnet_hvd.py \
                          --data_dir $JOB_DIR --num_epochs 90 -b $BATCH_SIZE \
                          --lr_decay_mode poly --warmup_epochs 10 --clear_log
  sleep 2

  #tar -czvf $JOB_DIR/batch_output_$AWS_BATCH_JOB_ID.tar.gz $SCRATCH_DIR/*
  #aws s3 cp $JOB_DIR/batch_output_$AWS_BATCH_JOB_ID.tar.gz $S3_OUTPUT

  log "done! goodbye, writing exit code to $AWS_BATCH_EXIT_CODE_FILE and shutting down my supervisord"
  echo "0" > $AWS_BATCH_EXIT_CODE_FILE
  kill  $(cat /tmp/supervisord.pid)
  exit 0
}


# Fetch and run a script
report_to_master () {
  # looking for masters nodes ip address calling the batch API TODO
  # get own ip and num cpus
  #
  ip=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)
  if [ -x "$(command -v nvidia-smi)" ] ; then
      NUM_GPUS=$(ls -l /dev/nvidia[0-9] | wc -l)
      availablecores=$NUM_GPUS
  else
      availablecores=$(nproc)
  fi
  log "I am a child node -> $ip:$availablecores, reporting to the master node -> ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}"
  until echo "$ip slots=$availablecores" | ssh ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS} "cat >> /$HOST_FILE_PATH"
  do
    echo "Sleeping 5 seconds and trying again"
  done
  log "done! goodbye"
  exit 0
  }


# Main - dispatch user request to appropriate function
log $NODE_TYPE
case $NODE_TYPE in
  main)
    wait_for_nodes "${@}"
    ;;

  child)
    report_to_master "${@}"
    ;;

  *)
    log $NODE_TYPE
    usage "Could not determine node type. Expected (main/child)"
    ;;
esac
