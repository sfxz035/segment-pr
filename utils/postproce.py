import numpy as np
import cv2 as cv



def connectComp(img):
    imgPre = np.greater(img, 200)
    imgPre = imgPre.astype(np.uint8)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(imgPre, connectivity=8)

    ####  滤除掉像素点极少的区域，输出区域数组
    rect_squence = []
    for i in range(ret-1):
        mask = (labels==i+1)
        area = stats[i+1][-1]
        if area > 30:
            rect_squence.append(mask)
    rect = np.asarray(rect_squence)
    return rect


def filterFewPoint(mask):
    imgPre = np.greater(mask, 200)
    imgPre = imgPre.astype(np.uint8)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(imgPre, connectivity=8)

    for i in range(ret-1):
        maskzj = (labels==i+1)
        area = stats[i+1][-1]
        if area < 25:
            labels[maskzj] = 0
        else:
            labels[maskzj] = 255
    return labels
def contourmask(img,mask,collections=None):
    maskFilt = filterFewPoint(mask)
    maskFilt = maskFilt.astype(np.uint8)
    contours, hierarchy = cv.findContours(maskFilt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    nub = len(contours)
    boxes = []
    for i in range(0, nub):
        x, y, w, h = cv.boundingRect(contours[i])
        boxes.append((x,y,x+w,y+h))
        cv.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 1)
    collections['boxes'] = boxes
    collections['nub'] = nub
