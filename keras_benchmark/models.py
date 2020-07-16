# step1
import cv2
import numpy as np
import time
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input

class ImageTools(object):
    """
    使用keras预训练模型进行图像识别
    """
    def __init__(self, img, model, w):
        self.image = img
        self.model = model
        self.weight = w

    # step2
    def image2matrix(self, img):
        """
        将图像转为矩阵
        """
        image = cv2.imread(img)
        #print (image.shape)
        image = cv2.resize(image, self.dim)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    @property
    def dim(self):
        """
        图像矩阵的维度
        """
        if self.model in ["inceptionv3", "inceptionresnetv2"]:
            shape = (299, 299)
        else:
            shape = (224, 224)

        return shape

    @property
    def Model(self):
        """
        模型
        """
        models = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "resnet50": ResNet50,
            "inceptionv3": InceptionV3,
            "inceptionresnetv2": InceptionResNetV2
        }

        return models[self.model]

    # step3
    def prediction(self,batch_size, num_img):
        """
        预测
        """
        model = self.Model(weights=self.weight)
        if self.model in ["inceptionv3", "inceptionresnetv2"]:
            preprocess = preprocess_input(self.image2matrix(self.image))
        else:
            preprocess = imagenet_utils.preprocess_input(self.image2matrix(self.image))


        # 把图片读取出来放到列表中
        res = []
        for i in range(batch_size):
            if self.model in ["inceptionv3", "inceptionresnetv2"]:
                preprocess = preprocess_input(self.image2matrix(self.image))
            else:
                preprocess = imagenet_utils.preprocess_input(self.image2matrix(self.image))

            res.append(preprocess)
        # 把图片数组联合在一起
        x = np.concatenate([x for x in res])
        start = time.time()
        predict = model.predict(x, batch_size = batch_size)
        end = time.time()
        yongshi = end - start
        #print("Total images is %s batch num: %s, total time cost: %.5f sec" % (str(num_img),str(batch_size),yongshi))
        return yongshi

if __name__ == "__main__":
    image = "../test.png"
    model = "vgg16"
    weight = "imagenet"
    batch_size =2
    num_img = 6
    tools = ImageTools(image, model, weight)
    tools.prediction(batch_size, num_img)
