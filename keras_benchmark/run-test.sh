#!/bin/bash
#run different model inference on cpu/gpu
#change model name to do different tests
python keras_benchmark/run_benchmark.py --model_name 'vgg19' --num_images 8 --gpu_cpu_id '-1' --save_path './keras_result_cpu.csv'
#python keras_benchmark/run_benchmark.py --model_name 'vgg19' --num_images 8 --gpu_cpu_id '0' --save_path './keras_result_gpu.csv'
