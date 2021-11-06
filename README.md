# Visually Significant Cataract AI Model Test
AI classification model for Visually Significant Cataract prediction

# Prerequisite

## Hardware Resource Recommendations
- CPU: Intel Core or Xeon Serial 64 bits Processors (released in recent years)
- Memory: More than 16G
- Disk: More than 20G free space
- GPU: Not necessary

## User
A sudo user is required for running commands in following sections.

## Operating System
Ubuntu 18.04 LTS (64 bits), Ubuntu 16.04 LTS (64 bits), Ubuntu 20.04 LTS or later versioin

#### /etc/apt/sources.list
```
# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.
deb http://archive.ubuntu.com/ubuntu/bionic main restricted
# deb-src http://archive.ubuntu.com/ubuntu/ bionic main restricted

## Major bug fix updates produced after the final release of the
## distribution.
deb http://archive.ubuntu.com/ubuntu/bionic-updates main restricted
# deb-src http://archive.ubuntu.com/ubuntu/bionic-updates main restricted

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team. Also, please note that software in universe WILL NOT receive any
## review or updates from the Ubuntu security team.
deb http://archive.ubuntu.com/ubuntu/bionic universe
# deb-src http://archive.ubuntu.com/ubuntu/bionic universe
deb http://archive.ubuntu.com/ubuntu/bionic-updates universe
# deb-src http://archive.ubuntu.com/ubuntu/bionic-updates universe

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team, and may not be under a free licence. Please satisfy yourself as to
## your rights to use the software. Also, please note that software in
## multiverse WILL NOT receive any review or updates from the Ubuntu
## security team.
deb http://archive.ubuntu.com/ubuntu/bionic multiverse
# deb-src http://archive.ubuntu.com/ubuntu/ bionic multiverse
deb http://archive.ubuntu.com/ubuntu/bionic-updates multiverse
# deb-src http://archive.ubuntu.com/ubuntu/bionic-updates multiverse

## N.B. software from this repository may not have been tested as
## extensively as that contained in the main release, although it includes
## newer versions of some applications which may provide useful features.
## Also, please note that software in backports WILL NOT receive any review
## or updates from the Ubuntu security team.
deb http://archive.ubuntu.com/ubuntu/bionic-backports main restricted universe multiverse
# deb-src http://archive.ubuntu.com/ubuntu/bionic-backports main restricted universe multiverse

## Uncomment the following two lines to add software from Canonical's
## 'partner' repository.
## This software is not part of Ubuntu, but is offered by Canonical and the
## respective vendors as a service to Ubuntu users.
# deb http://archive.canonical.com/ubuntu bionic partner
# deb-src http://archive.canonical.com/ubuntu bionic partner

deb http://security.ubuntu.com/ubuntu/bionic-security main restricted
# deb-src http://security.ubuntu.com/ubuntu/bionic-security main restricted
deb http://security.ubuntu.com/ubuntu/bionic-security universe
# deb-src http://security.ubuntu.com/ubuntu/bionic-security universe
deb http://security.ubuntu.com/ubuntu/bionic-security multiverse
# deb-src http://security.ubuntu.com/ubuntu/bionic-security multiverse
```
#### System should be updated to latest version:
```
sudo apt-get update
sudo apt-get upgrade -y
```

## Software
#### Reqired System Software Packages
Recommend to use python3.7 environment.
```
sudo apt-get install -y python3.7 python-pip python3.7-tk tk-dev build-essential swig libsm6 libxrender1 libxext-dev
```
pip recommend to be upgraded to latest version:
```
pip install --upgrade pip
```
If this is the 1st time to upgrade pip as normal user, logout and login will be required in order to use the new version **pip** installed in user home directory.

#### Required Python Packages
All required packages with specific versions are listed in file **requirements.txt**, run command to install:
```
pip install -r requirements.txt
```
You can also use virtual environment to setup the working environment, run command to install:
```
sudo apt-get install python3-venv
apt-get install python3.7-dev python3.7-venv
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
Copy all testing fundus image files into one folder, e.g. **./images**. The supported image file format are: png, jpg, or tiff.

# Prediction

## Usage
```
usage: python3.7 main.py --input DATASET_DIR [--output OUTPUT_DIR] [--threshold THRESHOLD_VALUE] [-h]

options:
  --input DATASET_DIR         The input directory for dataset image files, must be specified.
  --output OUTPUT_DIR         The result output csv file directory, optional, default to *./outputs*.
  -h                          Show command line options.

examples:
  git clone https://github.com/SunnyAVT/visually_significant_cataract.git
  cd visually_significant_cataract
  python3 main.py
  python3 main.py --input ./images --output ./outputs 
```

## Result
The prediction result will be shown at the end of the program stdout. The result will be also stored in a file with name **TestResult.csv** in **OUTPUT_DIR**.

## Example
There are 4 sample images under ./images directory for demo purpose:
```
> python3.7 main.py --input ./images --output ./outputs
Using Theano backend.
loading model: paper_retina_ref_048_Resnet50_cataract_model_p3.dat
loading paper_retina_ref_048_Resnet50_cataract_model_p3.dat model, time 0.03

100%|##########################################################| 4/4 [00:10<00:00,  3.30s/it]
filename:['201699522_L.jpg'], probability:0.4039306938648224
filename:['201696189_R.jpg'], probability:0.14523939788341522
filename:['201696189_L.jpg'], probability:0.05956516042351723
filename:['201698588_R.jpg'], probability:0.1625579446554184
Cataract Test is Over, Get your results in outputs/TestResult.csv !!!
```
Note: You can replace the images in ./images directory with your test retinal fundus images, then run the same command as above for the AI cataract prediction.

# Trouble shooting
1) Because of the difference in the OS, installation package or running environment, user could encounter error in setup
```
pip install -r requirements.txt
```
You can use try different packages module version as below.
Such as, you can use tensorflow==1.15 to replace tensorflow==1.14.0
Such as, you can use numpy==1.20.1 to replace numpy==1.19.5

2) Don’t need to care about the warning message in the running, it will not affect the final results. 
The reason is that the code stick to use tensorflow 1.x version in order to compatible with some AI libs.
```
resnet50.py:60: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(2048, (1, 1), name="res5b_branch2c")`
  x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
resnet50.py:51: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(512, (1, 1), name="res5c_branch2a")`
  x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
resnet50.py:56: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(512, (3, 3), name="res5c_branch2b", padding="same")`
  border_mode='same', name=conv_name_base + '2b')(x)
resnet50.py:60: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(2048, (1, 1), name="res5c_branch2c")`
```

# Citation
Please cite our paper if you find this code useful in your research. The BibTeX entry for the paper is:
APA
```
Tham, Y. C., Goh, J. H. L., Anees, A., Lei, X, et al., Detecting Visually Significant Age-Related Cataract using Retinal Photograph-Based Deep Learning: Development, Validation, and Comparison with Clinical Experts (Version 2.0.4) [Computer software]. https://doi.org/10.xxx/zenodo.xxxx
```
BibTeX
```
@software{Tham_Detecting_Visually_Significant,
author = {Tham, Yih Chung and Goh, Jocelyn Hui Lin and Anees, Ayesha and Lei, Xiaofeng, et al.},
doi = {10.5281/zenodo.xxxx},
title = {{Detecting Visually Significant Age-Related Cataract using Retinal Photograph-Based Deep Learning: Development, Validation, and Comparison with Clinical Experts}},
url = {https://github.com/SunnyAVT/visually_significant_cataract},
version = {2.0.4}
}
```
