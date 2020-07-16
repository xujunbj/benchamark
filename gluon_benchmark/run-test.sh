#!/bin/bash
#run different model inference on cpu/gpu
#change model name to do different tests

python gluon_benchmark/run_benchmark.py --model_name 'vgg19' --num_images 8 --mode 'cpu' --save_path './gluon_result_cpu.csv'
#python gluon_benchmark/run_benchmark.py --model_name 'resnet50' --num_images 1024 --mode 'gpu' --save_path './gluon_result_gpu.csv'
