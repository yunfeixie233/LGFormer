
# SPFormer-V1: Semantic Segmentation Repository

This repository is currently under construction and hosts an unofficial implementation of Superpixel Transformers for Efﬁcient Semantic Segmentation (SPFormer-V1). SPFormer-V1 project provides an efficient implementation of the semantic segmentation model with robust performance.

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
git clone https://github.com/SheldonXie233/SPFormer-V1.git
cd SPFormer
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
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
## Demo
To modify the model without the need to debug through the complex MMsegmentation framework, use the following command:

```python
python ./demo.py
```

## Training

To train the model, use the following command:

```bash
bash ./tools/dist_train.sh ./configs/stvit/stvit_r50_ade20k-512x512.py 1
```

## Testing

To test the model, use the following command. Replace `${CHECKPOINT_FILE}` with the path to your saved model weights.

```bash
bash ./tools/dist_test.sh ./configs/stvit/stvit_r50_ade20k-512x512.py ${CHECKPOINT_FILE}
```

For additional details on training and testing, please refer to the official [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html).

