import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import itk
# %matplotlib inline

# path
def load(file_path):
    # read all data
    data=open(file_path,'rb').read()
    # del the head part
    image_data = data[320:]
    # get image size
    image_size = np.sqrt((len(image_data) / 2)).astype(int).tolist()
    # define the image
    image = np.empty((image_size, image_size), dtype=float)
    # loop for insert data
    for i in range(image_size):
        for j in range(image_size):
            index = i * image_size + j
            val = int(image_data[2 * index + 1]) * 256 + int(image_data[2 * index])
            image[j, i] = val
    image = image.astype(np.float32)

    # nomarlize
    # image_std = (image - np.mean(image)) / np.std(image)
    # image_std_clip = np.clip(image_std, -0.75, 0.75)
    ## 添加通道，  映射到0，255
    # image_minmax = (image_std_clip-np.min(image_std_clip))/(np.max(image_std_clip)-np.min(image_std_clip))
    # img = (image_minmax*255).astype(np.uint8)
    # img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    ## 映射到0，1
    image_maxmin = (image-np.min(image))/(np.max(image)-np.min(image))

    ## write itk
    # image_itk = itk.GetImageFromArray(image)
    # # index
    # start = itk.Index[2]()
    # start.Fill(0)
    # # size
    # size = image_size
    # # region
    # region = itk.ImageRegion[2]()
    # region.SetSize([1024,1024])
    # region.SetIndex(start)
    # # set region
    # image_itk.SetRegions(region)
    # # set origin
    # image_itk.SetOrigin([0,0])
    # image_itk.SetSpacing([1,1])
    # itk.NiftiImageIOFactory.RegisterOneFactory
    # writer = itk.ImageFileWriter[itk.Image[itk.F, 2]].New()
    # writer.SetInput(image_itk)
    # writer.SetFileName('C:\\Users\\Administrator\\Desktop\\1.nii')
    # writer.Update()
    print('convert done')

    return image

if __name__ == '__main__':
    file_path = 'C:\\Users\\Administrator\\Desktop\\liver_cases_rawdata\\liver_cases_rawdata\\yanjie\\B_13_LI_1498536598_843000_UNPROCESSED_IBRST_00'
    a = load(file_path)
    a = cv.flip(a,0,dst=None)
    plt.imshow(a, cmap=plt.cm.gray)
    plt.show()
