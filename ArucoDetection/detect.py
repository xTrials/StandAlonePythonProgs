import numpy as np
import cv2
import cv2.aruco as aruco

# OpenCV v3.4.8
image = cv2.imread('ArucoImage.png', cv2.IMREAD_UNCHANGED)

# get dimensions of image
dimensions = image.shape

# height, width, number of channels in image
height = image.shape[0]
width = image.shape[1]

# print out the image parameters
print('Image Dimension    : ', dimensions)
print('Image Height       : ', height)
print('Image Width        : ', width)

# scaling the output image window
window_scale = 3
height_scaled = int(height/window_scale)
width_scaled = int(width/window_scale)

# init the dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# what we are expecting to see, arcuo marker 9,19,120
marker9 = aruco.drawMarker(aruco_dict, 9, 200, 1)
marker19 = aruco.drawMarker(aruco_dict, 19, 200, 1)
marker120 = aruco.drawMarker(aruco_dict, 120, 200, 1)


parameters = aruco.DetectorParameters_create()
# parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
# parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
# parameters.adaptiveThreshConstant = 7
# parameters.adaptiveThreshWinSizeMax = 500
# parameters.adaptiveThreshWinSizeMin = 100
parameters.adaptiveThreshWinSizeStep = 1
# parameters.cornerRefinementMaxIterations = 100
# parameters.cornerRefinementWinSize = 1
# parameters.cornerRefinementMinAccuracy = 0.01
# parameters.errorCorrectionRate = 0.6
# parameters.minCornerDistanceRate = 0.1
# parameters.markerBorderBits = 1
# parameters.maxErroneousBitsInBorderRate = 0.04
# parameters.minDistanceToBorder = 3
# parameters.minOtsuStdDev = 5
# parameters.minMarkerPerimeterRate = 0.01
# parameters.maxMarkerPerimeterRate = 4
# parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
# parameters.perspectiveRemovePixelPerCell = 8
# parameters.polygonalApproxAccuracyRate = 0.01
# parameters.perspectiveRemovePixelPerCell = 4
# parameters.errorCorrectionRate = 0.9
# parameters.aprilTagQuadSigma = 0.5
# parameters.aprilTagMinClusterPixels = 5
# parameters.aprilTagMaxLineFitMse = 20
# parameters.aprilTagMaxNmaxima = 20
# parameters.aprilTagMinWhiteBlackDiff = 40
# parameters.polygonalApproxAccuracyRate = 0.5

image1 = image
corners, ids, rejectedImgPoints = aruco.detectMarkers(
    image1, aruco_dict, parameters=parameters)

aruco_image = aruco.drawDetectedMarkers(
    image1, corners, ids, borderColor=(0, 0, 255))


image2 = image
image_discarded = aruco.drawDetectedMarkers(
    image=image2, corners=rejectedImgPoints, borderColor=(255, 0, 0))


# cv2.imshow('marker9', marker9)
# cv2.imshow('marker19', marker19)
# cv2.imshow('marker120', marker120)

cv2.namedWindow('ArucoImageDisplay', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ArucoImageDisplay', (height_scaled, width_scaled))
cv2.imshow('ArucoImageDisplay', aruco_image)

cv2.namedWindow('RejectedImages', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RejectedImages', (height_scaled, width_scaled))
cv2.imshow('RejectedImages', image_discarded)


cv2.waitKey(0)
cv2.destroyAllWindows()
