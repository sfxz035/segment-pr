import tensorflow as tf
import os
import numpy as np
import networks.U_net as U_net
import cv2 as cv
from utils import postproce
import utils.writeXml as wrx
import read
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()
savepath = '.\libSaveNet\save_unet3\conv_unet59999.ckpt-done'
data_dir = './picture/'
write_dir='./output/'
collection = {}
channel = 1
def output():
    file = os.listdir(data_dir)
    filesp = file[0].split('.')
    imgwritpath = write_dir+file[0]
    writepath = write_dir+filesp[0]+'.xml'
    collection['file'] = file[0]
    read_path = data_dir+file[0]

    ## png图片
    # img = cv.imread(read_path)

    # ###  原图
    img = read.load(read_path)
    img = cv.flip(img, 0, dst=None)


    ## 1 归一化到0，255的数据
    # image= img/255
    ## 2 截取到-0.75，0.75区间的数据/3
    img = np.expand_dims(img,-1)

    imgShape = np.shape(img)
    collection['shape'] = imgShape
    inputTest = np.expand_dims(img,0)
    x = tf.placeholder(tf.float32,shape = [1,imgShape[0],imgShape[1], channel])
    y = U_net.inference(x,is_training=False)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    output = sess.run(y, feed_dict={x: inputTest})

    out = np.squeeze(output).astype(np.uint8)


    ## 1 归一化到0，255的数据
    # out = out*255
    ## 2 截取到-0.75，0.75区间的数据
    # img = ((np.squeeze(img) + 0.75) / 1.5 * 255).astype(np.uint8)
    # out = out * 255
    ## 3 映射到0，1的数据
    img = ((np.squeeze(inputTest)) * 255).astype(np.uint8)
    out = np.squeeze(output).astype(np.uint8)
    out = out * 255


    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 定义结构元素
    outclosing = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel)  # 闭运算
    postproce.contourmask(img, outclosing,collection)
    wrx.writeInfoToXml(writepath,collection)
    # cv.imwrite(imgwritpath,img)
    cv.namedWindow('imgrec', 0)
    cv.resizeWindow('imgrec', 500, 500)
    cv.imshow('imgrec', img)
    cv.namedWindow('output', 0)
    cv.resizeWindow('output', 500, 500)
    cv.imshow('output', outclosing)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    output()