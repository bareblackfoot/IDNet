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
    STEPSIZE="[50000]"
    ITERS0=40000
    ITERS1=70000
    ITERS2=100000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS0=70000
    ITERS1=110000
    ITERS2=130000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_val"
    STEPSIZE="[350000]"
    ITERS0=360000
    ITERS1=490000
    ITERS2=510000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_idnet_iter_${ITERS2}.ckpt
  TAG=${EXTRA_ARGS_SLUG}
else
LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_idnet_iter_${ITERS2}.ckpt
  TAG="default"
fi
set -x
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
  --weight data/imagenet_weights/${NET}.ckpt \
  --imdb ${TRAIN_IMDB} \
  --imdbval ${TEST_IMDB} \
  --iters ${ITERS0} \
  --cfg experiments/cfgs/${NET}.yml \
  --mode "FRCNN" \
  --net ${NET} \
  --tag ${TAG} \
  --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
  TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
  --weight output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_idnet_iter_${ITERS0}.ckpt \
  --imdb ${TRAIN_IMDB} \
  --imdbval ${TEST_IMDB} \
  --iters ${ITERS1} \
  --cfg experiments/cfgs/${NET}_idn_${DATASET}.yml \
  --mode "QUAL" \
  --net ${NET} \
  --tag ${TAG} \
  --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
  TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
./experiments/scripts/test_nms.sh $@
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
  --weight output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_idnet_iter_${ITERS1}.ckpt \
  --imdb ${TRAIN_IMDB} \
  --imdbval ${TEST_IMDB} \
  --iters ${ITERS2} \
  --cfg experiments/cfgs/${NET}_idn_${DATASET}.yml \
  --mode "SIM" \
  --net ${NET} \
  --tag ${TAG} \
  --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
  TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
./experiments/scripts/test_idnet.sh $@