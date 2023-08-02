
# SPFormer-V1: Semantic Segmentation Repository

This repository is currently under construction and hosts an unofficial implementation of Superpixel Transformers for Efﬁcient Semantic Segmentation (SPFormer-V1). SPFormer-V1 project provides an efficient implementation of the semantic segmentation model.

## Prerequisites

- mmsegmentation 1.0.0 
- pytorch 1.8.1

## Installation

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
**Step 1.** Clone the Repository

To get started, clone the repository to your local machine with the following command:

```shell
git clonehttps://github.com/yunfeixie233/SpformerV1.git
cd SpformerV1
pip install -v -e .
pip install -r requirements.txt
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
cd mmpretrain
pip install -v -e .
# mmpretrain library for data augmentation
```


## Dataset Preparation

The ADE20K dataset is used for training and validation, which can be downloaded [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). The file structure of ADE20K is as follows:

<provide file structure here>

```
    ├── data                                               
    │   ├── ade/ADEChallengeData2016                                   
    │   │   ├── annotations                                     
    │   │   │   ├── training                                                                               
    │   │   │   ├── validation                                    
    │   │   ├──  images                                
    │   │   │   ├── training                                                                          
    │   │   │   ├── validation                                                              
    
```

## Training

To train the model, use the following command:

```bash
python ./tools/train.py ./configs/superformer/superformer_reshape_1stage.py
```

For the specific meaning in config setting, you can refer to https://mmsegmentation.readthedocs.io/en/latest/user_guides/1_config.html

## Testing

To test the model, use the following command. Replace `${CHECKPOINT_FILE}` with the path to your saved model weights.

```bash
python ./tools/test.py ./configs/superformer/superformer_reshape_1stage.py  ${CHECKPOINT_FILE}
```

For additional details on training and testing, please refer to the official [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html).

## Model Path

The model is added and registered in MMSegmentation.

- Backbone: `Resnet-50` with feature fusion from different stage.
- Head: `SuperformerBottleNeck` to finish Superpixel Tokenization, Classification and Superpixel Association

You can find the implementation of the model in the following directory:

```
SpformerV1
|__mmseg
   |__models
      |__backbones
         |__resnet.py
      |__decode_heads
         |__SuperformerBottleNeck.py
```

The original Superformer modules and ops have been put under the following directory:

```
mmseg/superformer
├── superpixel
│   ├── dual_path_transformer_ops.py
│   ├── superpixel_cross_attention_new_ops.py
│   ├── superpixel_cross_attention_ops.py
│   ├── superpixel_ops.py
│   ├── superpixel_transformer.py
│   └── test_superpixel_transformer_ops.py
```
