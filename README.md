# IDNet
A Tensorflow implementation of IDNet[Learning Instance-Aware Object Detection Using Determinantal Point Processes](https://arxiv.org/pdf/1805.10765.pdf) by Nuri Kim (nuri.kim@cpslab.snu.ac.kr). This repository is based on the tensorflow implementation of Faster R-CNN available [here](https://github.com/endernewton/tf-faster-rcnn). 

### Detection Performance
The current code supports **VGG16** model.

With VGG16 (``conv5_3``):
  - Train on VOC 2007 trainval and test on VOC 2007 test, **72.2**.
  - Train on VOC 2007+2012 trainval and test on VOC 2007 test, **76.8**.
  - Train on COCO 2014 train set and test on validation set, **27.3**.

### Prerequisites
  - A basic Tensorflow installation. The code follows **r1.2** format. If you are using r1.0, please check out the r1.0 branch to fix the slim Resnet block issue. If you are using an older version (r0.1-r0.12), please check out the r0.12 branch. While it is not required, for experimenting the original RoI pooling (which requires modification of the C++ code in tensorflow), you can check out my tensorflow [fork](https://github.com/endernewton/tensorflow) and look for ``tf.image.roi_pooling``.
  - Python packages you might not have: `cython`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. I use 1.6.
  - Docker users: Since the recent upgrade, the docker image on docker hub (https://hub.docker.com/r/mbuckler/tf-faster-rcnn-deps/) is no longer valid. However, you can still build your own image by using dockerfile located at `docker` folder (cuda 8 version, as it is required by Tensorflow r1.0.) And make sure following Tensorflow installation to install and use nvidia-docker[https://github.com/NVIDIA/nvidia-docker]. Last, after launching the container, you have to build the Cython modules within the running container. 

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/bareblackfoot/IDNet.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd IDNet/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

4. Install the [Python COCO API](https://github.com/pdollar/coco). The code requires the API to access COCO dataset.
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```

### Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets (Part of COCO is done). The steps involve downloading data and optionally creating soft links in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

If you find it useful, the ``data/cache`` folder created on my side is also shared [here](http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/cache.tgz).

### Demo and Test with pre-trained models
1. Download pre-trained model
  - Onedrive 
  [COCO](https://mysnu-my.sharepoint.com/:u:/g/personal/blackfoot_seoul_ac_kr/EbNEwAHsDulJpPq98xOqDs0BXfrXaC1k9QjsqjFbJlFImA?e=khRdbe).
  [VOC07](https://mysnu-my.sharepoint.com/:u:/g/personal/blackfoot_seoul_ac_kr/EVQkq2R3HAdOk3V4KVX7pmEB7kBCIX1HYKQNlo_O-3UzXg?e=GjodNh).
  [VOC0712](https://mysnu-my.sharepoint.com/:u:/g/personal/blackfoot_seoul_ac_kr/Ediu1LNBHs1ElWjaozh_ShMBledE39LIHjoQB6O5t74xVQ?e=Xg3W2y).

2. Create a folder and a soft link to use the pre-trained model
  ```Shell
  NET=vgg16
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  cd ../../..
  ```

3. Demo for testing on custom images
  ```Shell
  # at repository root
  GPU_ID=0
  CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
  ```
  
4. Test with pre-trained vgg16 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_idn.sh ${GPU_ID} pascal_voc vgg16
  ```

### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_idn.sh 0 pascal_voc vgg16
  ./experiments/scripts/train_idn.sh 1 coco vgg16
  ```
  
3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train/ --port=7002 &
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_idn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_idn.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_idn.sh 1 coco res101
  ```

5. You can use ``tools/reval.sh`` for re-evaluation


By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

The default number of training iterations is kept the same to the original Faster R-CNN for PASCAL VOC and COCO. 

### Citation
If you find this paper helpful, please consider citing:
    
    @article{kim2018learning,
        Author = {Nuri Kim and Donghoon Lee and Songhwai Oh},
        Title = {Learning Instance-Aware Object Detection Using Determinantal Point Processes},
        Journal = {arXiv preprint arXiv:1805.10765},
        Year = {2018}
    }
