---
## Installation
#### 1. System requirements
We run hmanet on a system running Ubuntu 18.01, with Python 3.6, PyTorch 1.8.1, and CUDA 10.1. For a full list of software packages and version numbers, see the Conda environment file `environment.yml`. 

This software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation. Thus, systems lacking a suitable GPU would likely take an extremely long time to train or evaluate models. The software was tested with the NVIDIA RTX 3090 GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory. 

#### 2. Installation guide
We recommend installation of the required packages using the conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, For use as integrative framework：
```
git clone https://github.com/cczu-CJL/HMANet.git
cd hmanet
conda env create -f environment.yml
source activate hmanet
pip install -e .
```

#### 3. Functions of scripts and folders
- **For evaluation:**
  - ``HMANet/hmanet/inference_acdc.py``
  
  - ``HMANet/hmanet/inference_synapse.py``
    
- **Data split:**
  - ``hmanet/hmanet/dataset_json/``
  
- **For inference:**
  - ``HMANet/hmanet/inference/predict_simple.py``
  
- **Network architecture:**
  - ``HMANet/hmanet/network_architecture/hmanet_acdc.py``
  
  - ``HMANet/hmanet/network_architecture/hmanet_synapse.py.py``
    
- **For training:**
  - ``HMANet/hmanet/run/run_training.py``
  
- **Trainer for dataset:**
  - ``HMANet/hmanet/training/network_training/hmanetTrainerV2_hmanet_acdc.py``
  
  - ``HMANet/hmanet/training/network_training/hmanetTrainerV2_hmanet_synapse.py.py``
  
---

## Training
#### 1. Dataset download
Datasets can be acquired via following links:

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)



#### 2. Setting up the datasets
After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./hmanet/
./DATASET/
  ├── hmanet_raw/
      ├── hmanet_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── hmanet_cropped_data/
  ├── hmanet_trained_models/
  ├── hmanet_preprocessed/
```
You can refer to ``HMANet/hmanet/dataset_json/`` for data split.

After that, you can preprocess the above data using following commands:
```
hmanet_convert_decathlon_task -i ../DATASET/hmanet_raw/hmanet_raw_data/Task01_ACDC
hmanet_convert_decathlon_task -i ../DATASET/hmanet_raw/hmanet_raw_data/Task02_Synapse

hmanet_plan_and_preprocess -t 1
hmanet_plan_and_preprocess -t 2
```

#### 3. Training and Testing
- Commands for training and testing:

```
bash train_inference.sh -c 0 -n hmanet_acdc -t 1 
#-c stands for the index of your cuda device
#-n denotes the suffix of the trainer located at HMANet/hmanet/training/network_training/
#-t denotes the task index
```
If you want use your own data, please create a new trainer file in the path ```hmanet/training/network_training``` and make sure the class name in the trainer file is the same as the trainer file. Some hyperparameters could be adjust in the trainer file, but the batch size and crop size should be adjust in the file```hmanet/run/default_configuration.py```.
 
- You can download our pretrained model weights via this [link](https://drive.google.com/drive/folders/1yvqlkeRq1qr5RxH-EzFyZEFsJsGFEc78?usp=sharing). Then, you can put model weights and their associated files in corresponding directories. For instance, on ACDC dataset, they should be like this:
```
../DATASET/hmanet_trained_models/hmanet/3d_fullres/Task001_ACDC/hmanetTrainerV2_hmanet_acdc__hmanetPlansv2.1/fold_0/model_best.model
../DATASET/hmanet_trained_models/hmanet/3d_fullres/Task001_ACDC/hmanetTrainerV2_hmanet_acdc__hmanetPlansv2.1/fold_0/model_best.model.pkl
```


#### 4. One Frequently Asked Problem
```
input feature has wrong size
```
If you encounter this problem during your implementation, please check the code in ``HMANet/hmanet/run/default_configuration.py``. I have set independent crop size (i.e., patch size) for each dataset. You may need to modify the crop size based on your own need.
