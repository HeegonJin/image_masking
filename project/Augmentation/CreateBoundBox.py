import cv2 as cv
import numpy as np


def thresh_callback(image, mask, val):
    threshold = val

    canny_output = cv.Canny(mask, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    mask_color = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,255,255]]
    for i in range(len(contours)):
        lst_intensities = []
        cimg = np.zeros_like(mask)
        cv.drawContours(cimg, contours_poly, i, color=255, thickness=-1)
        pts = np.where(cimg == 255)
        lst_intensities.append(mask[pts[0],pts[1]])

        color = mask_color[int(np.mean(lst_intensities).round())]
        cv.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])), \
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

    return image


def processImage(image, mask, thresh):
    if (image is None) or (mask is None):
        print('Could not open or find the image or mask file:')
        exit(0)
    bbox_mask = thresh_callback(image, mask, thresh)
    return bbox_mask
