""" Main entry point for running benchmarks with different Keras pretrained models."""
import sys
sys.path.append('./')
from keras_benchmark.models import ImageTools
import argparse
import os
import pandas as pd
import math

def recoginze_init_args():
    """
    initialize args
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        help='The benchmark can be run on cpu, gpu and multiple gpus.',
                        default= 'cpu')
    parser.add_argument('--model_name',
                        help='The model name to inference on, can be vgg16, vgg19, resnet50, inceptionv3, inceptionresnetv2',
                        default='vgg16')
    parser.add_argument('--image_path',
                        help='The model name to inference on, can be vgg16, vgg19, resnet50, inceptionv3, inceptionresnetv2',
                        default='./test.png')
    parser.add_argument('--model_weight',
                        help='The weight generated from',
                        default='imagenet')
    parser.add_argument('--num_images',
                        help='number of repeat time images to inference on',
                        default=4)
    parser.add_argument('--gpu_cpu_id',
                        required=True, help="-1 for cpu, 0 or 0,1.. for GPU")
    parser.add_argument('--save_path',
                        required=True, help="save path for result csv")

    return parser.parse_args()

def power_two(num):
    return num&num-1==0

if __name__ == '__main__':
    args = recoginze_init_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_cpu_id

    num_images = int(args.num_images)
    #vgg 19
    batch_size_ls = []
    time_ls = []

    print ("start inference!")
    n = int(math.log(num_images,2))
    for i in range(1,n+1):
        batch_size = int(math.pow(2,i))
        print ("process for batch %i"%(batch_size))
        tools = ImageTools(args.image_path, args.model_name, args.model_weight)
        ts = tools.prediction(i, int(args.num_images))
        batch_size_ls.append(batch_size)
        ts_update = ts/batch_size*num_images
        time_ls.append(ts_update)

    print ("start writing result!")

    #save result table into csv
    result = pd.DataFrame({'time(s)':time_ls,'batch_size':batch_size_ls})
    result['num_images'] = args.num_images
    if args.gpu_cpu_id=='-1':
        result['mode'] = "cpu"
    if args.gpu_cpu_id=='0':
        result['mode'] = "gpu"
    result['model_name'] = args.model_name

    result.to_csv(args.save_path)
    print ("FINISH")

