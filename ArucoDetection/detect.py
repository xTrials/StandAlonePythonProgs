import numpy as np
import math
import cv2
import cv2.aruco as aruco


def ImShowArucoMarkers(ids, dict):
    for id in range(len(ids)):
        marker_img = aruco.drawMarker(dict, id, 200, 1)
        cv2.imshow(str(id), marker_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def DetectArucos(img_names, do_display):
    ids_per_image = []
    for img_idx in range(len(img_names)):
        img_name = img_names[img_idx]

        print('loading image ', img_name)
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        height, width = image.shape
        # print('Image Height       : ', height)
        # print('Image Width        : ', width)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            image, aruco_dict, parameters=parameters)
        image_aruco = aruco.drawDetectedMarkers(
            image, corners, ids, borderColor=(0, 0, 255))
        image_discarded = aruco.drawDetectedMarkers(
            image=image, corners=rejectedImgPoints, borderColor=(255, 0, 0))

        ids = ids.flatten()
        ids_per_image.append(ids)
        # print(ids, end = " ")

        if(do_display):
            # scaling the output image window, useful for larger images
            window_scale = 3
            height_scaled = int(height/window_scale)
            width_scaled = int(width/window_scale)

            cv2.namedWindow('ImageAruco', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('ImageAruco', (height_scaled, width_scaled))
            cv2.imshow('ImageAruco', image_aruco)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return np.array(ids_per_image, dtype=object)


def CompareIds(expected_ids, detected_ids):
    print('expected_ids')
    for i in range(len(expected_ids)):
        sorted_arr = np.sort(expected_ids[i])
        expected_ids[i] = sorted_arr
        print(i, expected_ids[i])

    print('detected_ids')
    for i in range(len(detected_ids)):
        sorted_arr = np.sort(detected_ids[i])
        detected_ids[i] = sorted_arr
        print(i, detected_ids[i])

    return np.array_equal(expected_ids, detected_ids)


    # create parmaters OpenCV v3.4.8
parameters = aruco.DetectorParameters_create()

# adaptiveThreshWinSizeMin: minimum window size for
# adaptive thresholding before finding contours.
# parameters.adaptiveThreshWinSizeMin = 20  # int 3

# adaptiveThreshWinSizeMax: maximum window size for adaptive
# thresholding before finding contours.
# parameters.adaptiveThreshWinSizeMax = 50  # int 23

# adaptiveThreshWinSizeStep: increments from
# adaptiveThreshWinSizeMin to
# adaptiveThreshWinSizeMax during the thresholding
# parameters.adaptiveThreshWinSizeStep = 2  # int 10

# adaptiveThreshConstant: constant for adaptive thresholding
# before finding contours.
# parameters.adaptiveThreshConstant = 7.0  # double 7

# aprilTagCriticalRad: Reject quads where pairs of edges have
# angles that are close to straight or close to 180 degrees.
# Zero means that no quads are rejected.
# parameters.aprilTagCriticalRad = 10*math.pi/180  # float 10*PI/180 radians

# aprilTagDeglitch: should the thresholded image be deglitched?
# Only useful for very noisy images.
# parameters.aprilTagDeglitch = 0  # int 0

# aprilTagMaxLineFitMse: When fitting lines to the contours, what is the maximum mean squared error allowed?
# This is useful in rejecting contours that are far from being quad shaped;
# rejecting these quads "early" saves expensive decoding processing.
# parameters.aprilTagMaxLineFitMse = 10.0  # float 10

# aprilTagMaxNmaxima: how many corner candidates to consider when segmenting a group of pixels into a quad.
# parameters.aprilTagMaxNmaxima = 10  # int 10

# aprilTagMinClusterPixels: reject quads containing too few pixels.
# parameters.aprilTagMinClusterPixels = 5  # int 5

# aprilTagMinWhiteBlackDiff: When we build our model of black & white pixels,
# we add an extra check that the white model must be (overall)
# brighter than the black model. How much brighter? (in pixel values, [0,255]).
# parameters.aprilTagMinWhiteBlackDiff = 5  # int 5

# aprilTagQuadDecimate: Detection of quads can be done on a lower-resolution image,
# improving speed at a cost of pose accuracy and a slight decrease in detection rate.
# Decoding the binary payload is still done at full resolution. (default 0.0)
# parameters.aprilTagQuadDecimate = 0.0  # float 0

# aprilTagQuadSigma: What Gaussian blur should be applied to the segmented image (used for quad detection?)
# Parameter is the standard deviation in pixels.
# Very noisy images benefit from non-zero values (e.g. 0.8).
# parameters.aprilTagQuadSigma = 0.0  # float 0

# cornerRefinementMaxIterations: maximum number of iterations for stop criteria of the corner refinement process
# parameters.cornerRefinementMaxIterations = 100  # int 30

# cornerRefinementMethod: corner refinement method:
# CORNER_REFINE_NONE, no refinement.
# CORNER_REFINE_SUBPIX, do subpixel refinement.
# CORNER_REFINE_CONTOUR use contour-Points,
# CORNER_REFINE_APRILTAG use the AprilTag2 approach.
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR  # int 0

# cornerRefinementMinAccuracy: minimum error for the stop cristeria
# of the corner refinement process.
# parameters.cornerRefinementMinAccuracy = 0.1  # double 0.1

# cornerRefinementWinSize: window size for the corner refinement process (in pixels).
# parameters.cornerRefinementWinSize = 5  # int 5

# detectInvertedMarker: to check if there is a white marker. In order to generate a "white"
# marker just invert a normal marker by using a tilde, ~markerImage.
# parameters.detectInvertedMarker = False  # bool False

# errorCorrectionRate error correction rate respect to the maximun error correction capability for each dictionary
# parameters.errorCorrectionRate = 0.85  # double 0.6

# markerBorderBits: number of bits of the marker border, i.e. marker border width.
# parameters.markerBorderBits = 1  # int 1

# maxErroneousBitsInBorderRate: maximum number of accepted erroneous bits in the border
# (i.e. number of allowed white bits in the border).
# Represented as a rate respect to the total number of bits per marker
# parameters.maxErroneousBitsInBorderRate = 0.5  # double 0.35

# maxMarkerPerimeterRate: determine maximum perimeter for marker contour to be detected.
# This is defined as a rate respect to the maximum dimension of the input image.
# parameters.maxMarkerPerimeterRate = 4.0  # double 4

# minCornerDistanceRate: minimum distance between
# corners for detected markers relative to its perimeter.
# parameters.minCornerDistanceRate = 0.1  # double 0.05

# minDistanceToBorder: minimum distance of any corner to the image border for detected markers (in pixels).
# parameters.minDistanceToBorder = 3  # int 3

# minMarkerDistanceRate: minimum mean distance beetween two marker corners to be considered similar,
# so that the smaller one is removed.
# The rate is relative to the smaller perimeter of the two markers.
# parameters.minMarkerDistanceRate = 0.05  # double 0.05

# minMarkerPerimeterRate: determine minimum perimeter for marker contour to be detected.
# This is defined as a rate respect to the maximum dimension of the input image.
# parameters.minMarkerPerimeterRate = 0.03  # double 0.03

# minOtsuStdDev: minimun standard deviation in pixels values during the decodification step to apply
# Otsu thresholding
# (otherwise, all the bits are set to 0 or 1 depending on mean higher than 128 or not)
# parameters.minOtsuStdDev = 5.0  # double 5

# perspectiveRemoveIgnoredMarginPerCell: width of the margin of pixels on each cell not considered
# for the determination of the cell bit.
# Represents the rate respect to the total size of the cell, i.e. perspectiveRemovePixelPerCell.
# parameters.perspectiveRemoveIgnoredMarginPerCell = 0.25  # double 0.13

# perspectiveRemovePixelPerCell: number of bits (per dimension)
# for each cell of the marker when removing the perspective.
# parameters.perspectiveRemovePixelPerCell = 4  # int 4

# polygonalApproxAccuracyRate: minimum accuracy during the polygonal approximation process
# to determine which contours are squares.
# parameters.polygonalApproxAccuracyRate = 0.06  # double 0.03

# init the dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# optional: visually confirm that the markers ids you are looking
# for in the image correspond to their dictionary iamges
expected_ids = np.array([9, 19])
# ImShowArucoMarkers(expected_ids, aruco_dict)

test_images = ['ArucoImage9.png', 'ArucoImage1.png', 'ArucoImage2.png',
               'ArucoImage3.png', 'ArucoImage5.png',
               'ArucoImage7.png', 'ArucoImage8.png']
detected_ids_per_image = DetectArucos(test_images, True)

expected_ids_per_image = np.full(
    [len(test_images), len(expected_ids)], expected_ids)

test_passed = CompareIds(expected_ids_per_image, detected_ids_per_image)

print('Result: ', test_passed)
