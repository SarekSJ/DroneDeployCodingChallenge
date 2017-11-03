import numpy as np
import pprint
import cv2
from matplotlib import pyplot as plt


def draw_unit_vectors(img2, rvec, tvec, dst_pts, imat):
    distortion = np.array([]) # openCV treats empty distortion arrays as just 0 filled arrays
    axis = np.float32([[1000,0,0], [0,1000,0], [0,0,-1000]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, imat, distortion)
    p1 = (int(dst_pts[0][0][0]), int(dst_pts[0][0][1]))

    img = draw(img2, p1, imgpts)
    return img

def draw(img, p1, imgpts):
    # draw the different axes
    img = cv2.line(img, p1, (imgpts[0][0][0], imgpts[0][0][1]), (255,0,0), 10)
    img = cv2.line(img, p1, (imgpts[1][0][0], imgpts[1][0][1]), (0,255,0), 10)
    img = cv2.line(img, p1, (imgpts[2][0][0], imgpts[2][0][1]), (0,0,255), 10)
    return img