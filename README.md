# DL-INFERENCE-BENCHMARK

DL-INFERENCE-BENCHMARK

## Quick Start

* first create a ec2 instance, P3 machine,exg

* activate your python environment 

```bash
###keras
source activate tensorflow_p36
pip install opencv-python==4.1.0.25

##gluoncv
source activate conda_mxnet_p36
pip install gluoncv
```


* clone `git clone xxxx` this repo into your running environment

* Then run in terminal,edit the shell script yourself based on template

```bash
###keras
sh keras_benchmark/run-test.sh

###gluon
sh gluon_benchmark/run-test.sh
```

## Support framework
* Keras
* Gluon

## Models
* resnet50 
* vgg16
* vgg19

## Instance 
* C5
* G4
* P3
* Inf1
* EIA
* aliyun

## support inference test on
* GPU
* CPU

