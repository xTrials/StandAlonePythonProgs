import numpy as np
import math
import cv2
import cv2.aruco as aruco

# OpenCV v3.4.8
image = cv2.imread('ArucoImage1.png', cv2.IMREAD_GRAYSCALE)

# height, width, number of channels in image
height, width = image.shape

# print out the image dimensions
print('Image Height       : ', height)
print('Image Width        : ', width)

# scaling the output image window, useful for larger images
window_scale = 3
height_scaled = int(height/window_scale)
width_scaled = int(width/window_scale)

# init the dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# what we are expecting to see, arcuo marker 9,19,120
marker9 = aruco.drawMarker(aruco_dict, 9, 200, 1)
marker19 = aruco.drawMarker(aruco_dict, 19, 200, 1)
marker120 = aruco.drawMarker(aruco_dict, 120, 200, 1)

# let opencv draw the markers that you are expecting, if any
cv2.imshow('marker9', marker9)
cv2.imshow('marker19', marker19)
cv2.imshow('marker120', marker120)

# create parmaters
parameters = aruco.DetectorParameters_create()

# adaptiveThreshWinSizeMin: minimum window size for
# adaptive thresholding before finding contours.
parameters.adaptiveThreshWinSizeMin = 3  # int 3

# adaptiveThreshWinSizeMax: maximum window size for adaptive
# thresholding before finding contours.
parameters.adaptiveThreshWinSizeMax = 23  # int 23

# adaptiveThreshWinSizeStep: increments from
# adaptiveThreshWinSizeMin to
# adaptiveThreshWinSizeMax during the thresholding
parameters.adaptiveThreshWinSizeStep = 10  # int 10

# adaptiveThreshConstant: constant for adaptive thresholding
# before finding contours.
parameters.adaptiveThreshConstant = 7.0  # double 7

# aprilTagCriticalRad: Reject quads where pairs of edges have
# angles that are close to straight or close to 180 degrees.
# Zero means that no quads are rejected.
parameters.aprilTagCriticalRad = 10*math.pi/180  # float 10*PI/180 radians

# aprilTagDeglitch: should the thresholded image be deglitched?
# Only useful for very noisy images.
parameters.aprilTagDeglitch = 0  # int 0

# aprilTagMaxLineFitMse: When fitting lines to the contours, what is the maximum mean squared error allowed?
# This is useful in rejecting contours that are far from being quad shaped;
# rejecting these quads "early" saves expensive decoding processing.
parameters.aprilTagMaxLineFitMse = 10.0  # float 10

# aprilTagMaxNmaxima: how many corner candidates to consider when segmenting a group of pixels into a quad.
parameters.aprilTagMaxNmaxima = 10  # int 10

# aprilTagMinClusterPixels: reject quads containing too few pixels.
parameters.aprilTagMinClusterPixels = 5  # int 5

# aprilTagMinWhiteBlackDiff: When we build our model of black & white pixels, we add an extra check that the white model must be (overall)
# brighter than the black model. How much brighter? (in pixel values, [0,255]).
parameters.aprilTagMinWhiteBlackDiff = 5  # int 5

# aprilTagQuadDecimate: Detection of quads can be done on a lower-resolution image, improving speed at a cost of pose accuracy and a slight decrease in detection rate.
# Decoding the binary payload is still done at full resolution. (default 0.0)
parameters.aprilTagQuadDecimate = 0.0  # float 0

# aprilTagQuadSigma: What Gaussian blur should be applied to the segmented image (used for quad detection?) Parameter is the standard deviation in pixels.
# Very noisy images benefit from non-zero values (e.g. 0.8).
parameters.aprilTagQuadSigma = 0.0  # float 0

# cornerRefinementMaxIterations: maximum number of iterations for stop criteria of the corner refinement process (default 30).
parameters.cornerRefinementMaxIterations = 30  # int 30

# cornerRefinementMethod: corner refinement method:
# CORNER_REFINE_NONE, no refinement.
# CORNER_REFINE_SUBPIX, do subpixel refinement.
# CORNER_REFINE_CONTOUR use contour-Points,
# CORNER_REFINE_APRILTAG use the AprilTag2 approach.
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_NONE  # int 0

# cornerRefinementMinAccuracy: minimum error for the stop cristeria
# of the corner refinement process.
parameters.cornerRefinementMinAccuracy = 0.1  # double 0.1

# ornerRefinementWinSize: window size for the corner refinement process (in pixels).
parameters.cornerRefinementWinSize = 5  # int 5

# detectInvertedMarker: to check if there is a white marker. In order to generate a "white"
# marker just invert a normal marker by using a tilde, ~markerImage.
parameters.detectInvertedMarker = False  # bool False

# errorCorrectionRate error correction rate respect to the maximun error correction capability for each dictionary
parameters.errorCorrectionRate = 0.6  # double 0.6

# markerBorderBits: number of bits of the marker border, i.e. marker border width.
parameters.markerBorderBits = 1  # int 1

# maxErroneousBitsInBorderRate: maximum number of accepted erroneous bits in the border (i.e. number of allowed white bits in the border).
# Represented as a rate respect to the total number of bits per marker
parameters.maxErroneousBitsInBorderRate = 0.35  # double 0.35

# maxMarkerPerimeterRate: determine maximum perimeter for marker contour to be detected.
# This is defined as a rate respect to the maximum dimension of the input image.
parameters.maxMarkerPerimeterRate = 4.0  # double 4

# minCornerDistanceRate: minimum distance between
# corners for detected markers relative to its perimeter.
parameters.minCornerDistanceRate = 0.05  # double 0.05

# minDistanceToBorder: minimum distance of any corner to the image border for detected markers (in pixels).
parameters.minDistanceToBorder = 3  # int 3

# minMarkerDistanceRate: minimum mean distance beetween two marker corners to be considered similar, so that the smaller one is removed.
# The rate is relative to the smaller perimeter of the two markers.
parameters.minMarkerDistanceRate = 0.05  # double 0.05

# minMarkerPerimeterRate: determine minimum perimeter for marker contour to be detected.
# This is defined as a rate respect to the maximum dimension of the input image.
parameters.minMarkerPerimeterRate = 0.03  # double 0.03

# minOtsuStdDev: minimun standard deviation in pixels values during the decodification step to apply Otsu thresholding
# (otherwise, all the bits are set to 0 or 1 depending on mean higher than 128 or not)
parameters.minOtsuStdDev = 5.0  # double 5

# perspectiveRemoveIgnoredMarginPerCell: width of the margin of pixels on each cell not considered for the determination of the cell bit. Represents the rate respect to the total size of the cell, i.e. perspectiveRemovePixelPerCell.
parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13  # double 0.13

# perspectiveRemovePixelPerCell: number of bits (per dimension) for each cell of the marker when removing the perspective.
parameters.perspectiveRemovePixelPerCell = 4  # int 4

# polygonalApproxAccuracyRate: minimum accuracy during the polygonal approximation process
# to determine which contours are squares.
parameters.polygonalApproxAccuracyRate = 0.03  # double 0.03

corners, ids, rejectedImgPoints = aruco.detectMarkers(
    image, aruco_dict, parameters=parameters)

image_aruco = aruco.drawDetectedMarkers(
    image, corners, ids, borderColor=(0, 0, 255))

image_discarded = aruco.drawDetectedMarkers(
    image=image, corners=rejectedImgPoints, borderColor=(255, 0, 0))

cv2.namedWindow('ImageAruco', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ImageAruco', (height_scaled, width_scaled))
cv2.imshow('ImageAruco', image_aruco)

cv2.namedWindow('ImageDiscarded', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ImageDiscarded', (height_scaled, width_scaled))
cv2.imshow('ImageDiscarded', image_discarded)

cv2.waitKey(0)
cv2.destroyAllWindows()
