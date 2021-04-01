# AI4cataract
AI classification model for cataract prediction

# Prerequisite

## Hardware Resource Recommendations
- CPU: Intel Core or Xeon Serial 64 bits Processors (released in recent years)
- Memory: More than 16G
- Disk: More than 20G free space
- GPU: Not necessary

## User
A sudo user is required for running commands in following sections.

## Operating System
Ubuntu 18.04 LTS (64 bits) or Ubuntu 16.04 LTS (64 bits)

#### /etc/apt/sources.list
```
# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.
deb http://archive.ubuntu.com/ubuntu/ bionic main restricted
# deb-src http://archive.ubuntu.com/ubuntu/ bionic main restricted

## Major bug fix updates produced after the final release of the
## distribution.
deb http://archive.ubuntu.com/ubuntu/ bionic-updates main restricted
# deb-src http://archive.ubuntu.com/ubuntu/ bionic-updates main restricted

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team. Also, please note that software in universe WILL NOT receive any
## review or updates from the Ubuntu security team.
deb http://archive.ubuntu.com/ubuntu/ bionic universe
# deb-src http://archive.ubuntu.com/ubuntu/ bionic universe
deb http://archive.ubuntu.com/ubuntu/ bionic-updates universe
# deb-src http://archive.ubuntu.com/ubuntu/ bionic-updates universe

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team, and may not be under a free licence. Please satisfy yourself as to
## your rights to use the software. Also, please note that software in
## multiverse WILL NOT receive any review or updates from the Ubuntu
## security team.
deb http://archive.ubuntu.com/ubuntu/ bionic multiverse
# deb-src http://archive.ubuntu.com/ubuntu/ bionic multiverse
deb http://archive.ubuntu.com/ubuntu/ bionic-updates multiverse
# deb-src http://archive.ubuntu.com/ubuntu/ bionic-updates multiverse

## N.B. software from this repository may not have been tested as
## extensively as that contained in the main release, although it includes
## newer versions of some applications which may provide useful features.
## Also, please note that software in backports WILL NOT receive any review
## or updates from the Ubuntu security team.
deb http://archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse
# deb-src http://archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse

## Uncomment the following two lines to add software from Canonical's
## 'partner' repository.
## This software is not part of Ubuntu, but is offered by Canonical and the
## respective vendors as a service to Ubuntu users.
# deb http://archive.canonical.com/ubuntu bionic partner
# deb-src http://archive.canonical.com/ubuntu bionic partner

deb http://security.ubuntu.com/ubuntu/ bionic-security main restricted
# deb-src http://security.ubuntu.com/ubuntu/ bionic-security main restricted
deb http://security.ubuntu.com/ubuntu/ bionic-security universe
# deb-src http://security.ubuntu.com/ubuntu/ bionic-security universe
deb http://security.ubuntu.com/ubuntu/ bionic-security multiverse
# deb-src http://security.ubuntu.com/ubuntu/ bionic-security multiverse
```
#### System should be updated to latest version:
```
sudo apt-get update
sudo apt-get upgrade -y
```

## Software
#### Reqired System Software Packages
```
sudo apt-get install -y python3.7 python-pip python3.7-tk tk-dev build-essential swig libsm6 libxrender1 libxext-dev
```
pip should be upgraded to latest version:
```
pip install --upgrade pip
```
If this is the 1st time to upgrade pip as normal user, logout and login will be required in order to use the new version **pip** installed in user home directory.

#### Required Python Packages
All required packages with specific versions are listed in file **requirements.txt**, run command to install:
```
pip install -r requirements.txt
```

## Dataset
Copy all testing fundus image files into one folder, e.g. **./images**. The supported image file format are: png, jpg, or tiff.

# Prediction

## Usage
```
usage: python3.7 main.py --input DATASET_DIR [--label LABEL_FILE] [--output OUTPUT_DIR] [--threshold THRESHOLD_VALUE] [-h]

options:
  --input DATASET_DIR         The input directory for dataset image files, must be specified.
  --label LABEL_FILE          The ground truth csv file, optional.
  --output OUTPUT_DIR         The result output csv file directory, optional, default to *./outputs*.
  --threshold THRESHOLD_VALUE The threshold value, optional.
  -h                          Show command line options.

examples:
  python3.7 main.py
  python3.7 main.py --input ./images --label ground_truth.csv --output ./outputs --threshold 0.013259
```

## Result
The prediction result will be shown at the end of the program stdout. The result will be also stored in a file with name **TestResult.csv** in **OUTPUT_DIR**.

## Example
There are 6 sample images under ./images directory for demo purpose:
```
> python3.7 main.py --input ./images --output ./outputs
Using Theano backend.
loading model: referable_0.48_16July_Resnet50.dat
loading 34 model, time 0.02
100%|###################################################################################| 6/6 [00:09<00:00,  1.54s/it]
filename:['201699522_L.jpg'], probability:0.403930693865
filename:['201699522_R.jpg'], probability:0.26423445344
filename:['201696189_R.jpg'], probability:0.145239412785
filename:['201696189_L.jpg'], probability:0.0595651604235
filename:['201698588_R.jpg'], probability:0.162557944655
filename:['201698588_L.jpg'], probability:0.128469750285
Cataract Test is Over, Get your results in outputs/TestResult.csv !!!
```
