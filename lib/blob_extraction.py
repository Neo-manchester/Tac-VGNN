import cv2
import numpy as np


def img_preprocess_mask(raw_img, erosion = False, resize_x = 256, resize_y = 256, kernel_size = 2, binary_threshold = 150, circle_x_bias=-1,
         circle_y_bias=1, circle_radius_bias=-15):

        if erosion == True:
          kernel = np.ones((kernel_size, kernel_size),np.uint8)  
          erosion = cv2.erode(raw_img, kernel, iterations = 1)

        else:
          erosion = raw_img

        erosion= cv2.resize(erosion, (resize_x, resize_y))
   
        rows, cols, channel = erosion.shape
        grayscale = cv2.cvtColor(erosion, cv2.COLOR_RGB2GRAY)
        ret, gray = cv2.threshold(grayscale, binary_threshold, 255, cv2.THRESH_BINARY)

        # Build a single channel image，cols and rows represent (x,y) of circle center
        whole_black = np.zeros((rows, cols), np.uint8)
        whole_black[:, :] = 0 ##set as pure black(0)
        black_circle = cv2.circle(whole_black, ((cols//2)+circle_x_bias, (rows//2)+circle_y_bias), (cols//2)+circle_radius_bias, (255), -1) ##set insert circle
        processed = cv2.bitwise_and(black_circle, gray)
        
        whole_white = np.zeros((rows, cols), np.uint8)
        whole_white[:, :] = 255 ##set as pure black(0)
        white_circle = cv2.circle(whole_white, ((cols//2)+circle_x_bias, (rows//2)+circle_y_bias), (cols//2)+circle_radius_bias, (0), -1)
        white_mask = cv2.bitwise_or(white_circle, processed)

        return processed, white_mask, erosion


def img_preprocess(raw_img, erosion = False, kernel_size = 2, resize_x=300, resize_y=300, binary_threshold = 150, circle_x_bias=-1,
         circle_y_bias=1, circle_radius_bias=-15):
        
        if erosion == True:
          kernel = np.ones((kernel_size, kernel_size),np.uint8)  
          erosion = cv2.erode(raw_img, kernel, iterations = 1)

        else:
          erosion = raw_img

        erosion= cv2.resize(erosion, (resize_x, resize_y))
        
        rows, cols, channel = erosion.shape
        grayscale = cv2.cvtColor(erosion, cv2.COLOR_RGB2GRAY)
        ret, gray = cv2.threshold(grayscale, binary_threshold, 255, cv2.THRESH_BINARY)

        # Build a single channel image，cols and rows represent (x,y) of circle center
        whole_black = np.zeros((rows, cols), np.uint8)
        whole_black[:, :] = 0 ##set as pure black(0)
        black_circle = cv2.circle(whole_black, ((cols//2)+circle_x_bias, (rows//2)+circle_y_bias), (cols//2)+circle_radius_bias, (255), -1) ##set insert circle
        processed = cv2.bitwise_and(black_circle, gray)

        return processed


def blob_detect(processed, minArea = 25, blobColor = 255, 
                minCircularity = 0.01, minConvexity = 0.01, 
                minInertiaRatio = 0.01, thresholdStep = 5,
                minDistBetweenBlobs = 3.0, minRepeatability = 3):
  
        # SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()

        # Area
        params.filterByArea = True
        params.minArea = minArea

        # Color
        params.filterByColor = True
        params.blobColor = blobColor

        # Circularity
        params.filterByCircularity = True
        params.minCircularity = minCircularity

        # Convexity
        params.filterByConvexity = True
        params.minConvexity = minConvexity

        # Inertia
        params.filterByInertia = True
        params.minInertiaRatio = minInertiaRatio

        # Step
        params.thresholdStep = thresholdStep #steps to go through from minthreshold to maxthreshold
        params.minDistBetweenBlobs = minDistBetweenBlobs #avoid overlapping blobs
        params.minRepeatability = minRepeatability  #how stable the blob is between different thresholds

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(processed)


        return keypoints


def get_nodes_pos(keypoints):
        
        node_coordinates = []
        node_pos = []
        node_num = len(keypoints)

        for i in range(node_num):
            node_coordinates.append(np.array(keypoints[i].pt))

        node_pos = np.array(node_coordinates)

        return node_pos