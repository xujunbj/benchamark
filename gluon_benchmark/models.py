# step1
from mxnet import nd, image
import mxnet as mx
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
import time
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model

#reference https://gluon-cv.mxnet.io/model_zoo/classification.html
models = {
    "resnet50": "ResNet50_v2",
    "vgg16":"VGG16",
    "vgg19":"VGG19"
}


class ImageToolsGluon(object):
    """
    使用gluon预训练模型进行图像识别
    """
    def __init__(self, img, model):
        self.pic = img
        self.model = model

    # step3
    def prediction(self,batch_size,gpu_cpu_id):
        """
        预测
        """
        if gpu_cpu_id == 'cpu':
            ctx = mx.cpu()
        if gpu_cpu_id == 'gpu':
            ctx=mx.gpu()

        # Load Model
        model_name = models[self.model]
        pretrained = True

        net = get_model(model_name, pretrained=pretrained)
        classes = net.classes
        net.collect_params().reset_ctx(ctx)
        net.hybridize()

        # Load Images
        img = image.imread(self.pic)
        images = [img]*batch_size

        # Transform
        a = transform_eval(images)
        b = nd.concat(*a,dim=0)
        y = b.copyto(ctx)

        #run inference
        start = time.time()
        predict = net(y)
        #verify it's running on gpu by print the result to see whether ndarray on gpu
        #print (predict)
        end = time.time()
        yongshi = end - start
        #print("batch num: %s, total time cost: %.5f sec" % (str(batch_size),yongshi))

        return yongshi

if __name__ == "__main__":
    a = "../test.png"
    model = "resnet50"
    tools = ImageToolsGluon(a, model)
    tools.prediction(batch_size=6,gpu_cpu_id='cpu')

