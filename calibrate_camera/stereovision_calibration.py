import numpy as np
import cv2 as cv
import glob
import scipy.io
import h5py

MATLAB = True
if MATLAB:
    params_file = './matlab/stereoParams.mat'
    mat = scipy.io.loadmat(params_file)
    cameraMatrixL = mat['cameraMatrix1']
    distL = mat['distCoeffs1'].flatten()
    cameraMatrixR = mat['cameraMatrix2']
    distR = mat['distCoeffs2'].flatten()
    rot = mat['rot']
    trans = mat['trans'].T
    frameSize = (1920, 1080)
    fundamentalMatrix = mat['fundamentalMatrix']
    essentialMatrix = mat['essentialMatrix']
    rectifyScale = 0.9
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(cameraMatrixL, distL,
                                                                               cameraMatrixR, distR,
                                                                               frameSize, rot, trans,
                                                                                   rectifyScale, (0, 0))

    result = {}
    result['left-right'] = {}
    result['left-right']['cameraMatrix1'] = cameraMatrixL
    result['left-right']['cameraMatrix2'] = cameraMatrixR
    result['left-right']['distCoeffs1'] = distL
    result['left-right']['distCoeffs2'] = distR

    result['left-right']['R'] = rot
    result['left-right']['T'] = trans
    result['left-right']['R1'] = rectL
    result['left-right']['R2'] = rectR
    result['left-right']['P1'] = projMatrixL
    result['left-right']['P2'] = projMatrixR
    result['left-right']['F'] = fundamentalMatrix
    result['left-right']['E'] = essentialMatrix
    result['left-right']['roi1'] = roi_L
    result['left-right']['roi2'] = roi_R
    result['left-right']['Q'] = Q
    result['left-right']['image_shape'] = [frameSize, frameSize]

else:

    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

    chessboardSize = (9,6)
    frameSize = (1920, 1080)
    src =  './data/selected_images'
    imagesLeft = sorted(glob.glob(src + '/left/*left*.jpg'))
    imagesRight = [img.replace('left', 'right') for img in imagesLeft]

    assert len(imagesLeft) == len(imagesRight) # Same number of images from two cameras
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 50
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = [] # 2d points in image plane.


    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:

            objpoints.append(objp)

            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv.imshow('img left', imgL)
            cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv.imshow('img right', imgR)
            cv.waitKey(1)


    cv.destroyAllWindows()

    ############## CALIBRATION #######################################################

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    cameraMatrixL = np.array([[1400.59, 0, 1030.9], [0, 1400.59, 526.489], [0, 0, 1]])
    distL = np.array([-0.1711, 0.0250, 0.0004, 0.0001, 0.0004])
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    cameraMatrixR = np.array([[1399.43, 0, 1059.85], [0, 1399.43, 545.848], [0, 0, 1]])
    distR = np.array([-0.1701, 0.0252, 0.0004, 0.0001, 0.0004])
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    result = {}
    result['left-right'] = {}
    result['left-right']['cameraMatrix1'] = newCameraMatrixL
    result['left-right']['cameraMatrix2'] = newCameraMatrixR
    result['left-right']['distCoeffs1'] = distL
    result['left-right']['distCoeffs2'] = distR

    # result = {}
    # result['left-right'] = {}
    # result['left-right']['cameraMatrix1'] = np.array([[1400.59, 0, 1030.9], [0, 1400.59, 526.489], [0, 0, 1]])
    # result['left-right']['cameraMatrix2'] = np.array([[1399.43, 0, 1059.85], [0, 1399.43, 545.848], [0, 0, 1]])
    # result['left-right']['distCoeffs1'] = np.array([-0.1711, 0.0250, 0.0004, 0.0001, 0.0004])
    # result['left-right']['distCoeffs2'] = np.array([-0.1701, 0.0252, 0.0004, 0.0001, 0.0004])
    ########## Stereo Vision Calibration #############################################

    flags = cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags = flags)


    ########## Stereo Rectification #################################################

    rectifyScale= 0.9
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

    result['left-right']['R'] = rot
    result['left-right']['T'] = trans
    result['left-right']['R1'] = rectL
    result['left-right']['R2'] = rectR
    result['left-right']['P1'] = projMatrixL
    result['left-right']['P2'] = projMatrixR
    result['left-right']['F'] = fundamentalMatrix
    result['left-right']['E'] = essentialMatrix
    result['left-right']['roi1'] = roi_L
    result['left-right']['roi2'] = roi_R
    result['left-right']['Q'] = Q
    result['left-right']['image_shape'] = [(widthL, heightL), (widthR, heightR)]

fname = '../dlc3D/prob_3d/camera_matrix/stereo_params.pickle'
import os
import pickle
if os.path.isfile(fname):
    print('Removing existed file!')
    os.remove(fname)
with open(fname, 'wb') as f:
    pickle.dump(result, f)
print(result)
