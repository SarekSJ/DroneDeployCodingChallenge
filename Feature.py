import numpy as np
import cv2
from matplotlib import pyplot as plt
import transforms3d
import math
from drawings import *

MIN_MATCH_COUNT = 10

def get_pose_and_coordinates(pattern_file, image_file, cam_matrix):
    img1 = cv2.imread(pattern_file, 0)  # queryImage
    img2 = cv2.imread(image_file, 1) # trainImage
    # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT object
    sift = cv2.xfeatures2d.SIFT_create()


    # Find the keypoints / descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        object_points = np.float32([[0,0,0],[0,h,0],[w,h,0],[w,0,0]])

        result, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, dst, cam_matrix, np.array([]))
        rvec, jacob = cv2.Rodrigues(rvec)
        # Get the euler angles
        eulers = transforms3d.euler.mat2euler(rvec)
        # Calculate x, y, and z distances
        distances = -np.matrix(rvec).T * np.matrix(tvec)

        # Draw vectors onto template in image
        img2 = draw_unit_vectors(img2, rvec, tvec, dst, cam_matrix)

        yaw_string = 'Yaw: ' + str(eulers[2])
        pitch_string = 'Pitch: ' + str(eulers[1])
        roll_string = 'Roll: ' + str(eulers[0])
        x_string = 'X: ' + str(distances.item(0))
        y_string = 'Y: ' + str(distances.item(1))
        z_string = 'Z: ' + str(distances.item(2))

        # Print values to console first
        print('--------' + image_file + '--------')
        print(yaw_string)
        print(pitch_string)
        print(roll_string)
        print(x_string)
        print(y_string)
        print(z_string)
        print('--------/' + image_file + '--------')


        # Display all the values as text over the image (colored white for readability)
        plt.text(40, 200, yaw_string, color='white')
        plt.text(40, 400, pitch_string, color='white')
        plt.text(40, 600, roll_string, color='white')
        plt.text(40, 800, x_string, color='white')
        plt.text(40, 1000, y_string, color='white')
        plt.text(40, 1200, z_string,color='white')
        plt.imshow(img2)
        plt.show()
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor=(0,255,0), # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask, # draw only inliers
                       flags=2)
