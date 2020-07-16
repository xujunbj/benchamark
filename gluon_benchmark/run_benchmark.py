""" Main entry point for running benchmarks with different gloncv pretrained models."""
import sys
sys.path.append('./')
from gluon_benchmark.models import ImageToolsGluon
import argparse
import pandas as pd
import math

def recoginze_init_args():
    """
    initialize args
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        help='The benchmark can be run on cpu or gpu',
                        required= True)
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
                        default="-1", help="-1 for cpu, 0 or 0,1.. for GPU")
    parser.add_argument('--save_path',
                        required=True, help="save path for result csv")

    return parser.parse_args()

def power_two(num):
    return num&num-1==0

if __name__ == '__main__':
    args = recoginze_init_args()

    num_images = int(args.num_images)
    #vgg 19
    batch_size_ls = []
    time_ls = []

    print ("start inference!")

    n = int(math.log(num_images,2))
    for i in range(1,n+1):
        batch_size = int(math.pow(2,i))
        print ("process for batch %i"%(batch_size))
        tools = ImageToolsGluon(args.image_path, args.model_name)
        ts = tools.prediction(batch_size=batch_size, gpu_cpu_id="cpu")
        batch_size_ls.append(batch_size)
        ts_update = ts/batch_size*num_images
        time_ls.append(ts_update)

    print ("start writing result!")
    #save result table into csv
    result = pd.DataFrame({'time(s)':time_ls,'batch_size':batch_size_ls})
    result['num_images'] = args.num_images
    result['mode'] = args.mode
    result['model_name'] = args.model_name

    result.to_csv(args.save_path)
    print ("FINISH")

