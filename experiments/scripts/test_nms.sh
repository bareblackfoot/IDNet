#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_val"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  LOG="experiments/logs/test_nms_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_idnet_iter_${ITERS}.ckpt
  TAG=${EXTRA_ARGS_SLUG}
else
  LOG="experiments/logs/test_nms_${NET}_${TRAIN_IMDB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_idnet_iter_${ITERS}.ckpt
  TAG="default"
fi
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
--imdb ${TEST_IMDB} \
--model ${NET_FINAL} \
--cfg experiments/cfgs/${NET}.yml \
--net ${NET} \
--test "NMS" \
--tag ${TAG} \
--set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      ${EXTRA_ARGS}