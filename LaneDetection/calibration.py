import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle

def camera_calibration(folder, nx, ny):
    '''
    find (x, y) locations of all corners using openCV findChessBoardCorners
    folder: directory of the calibration images
    nx: expected number of corners along the x direction
    ny: expected number of corners along the y direction
    return a dictionary:
        ret: RMS Error of the calibration
        mtx: the camera matrix
        dist: distorsion coefficients
        rvecs: rotation vectors
        tvecs: translation vectors
    '''
    # Store object points and image points from all the images
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane
    #Prepare object points, like (0,0,0), (1, 0,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x, y coordinate
        
    assert len(folder) != 0, 'No file found in folder'
        
    for fname in folder:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        img_sz = gray.shape[::-1]
            
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    if len(objpoints) == len(imgpoints) and len(objpoints) != 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_sz, None, None)
        return {'ret': ret, 'cameraMatrix': mtx, 'distorsionCoeff': dist, \
                    'rotationVec': rvecs, 'translationVec': tvecs}
    else:
        raise Error('Camera Calibration failed')


nx = 9 #number of corners in a row
ny = 6 #numbers or corners in a column
folder_calibration = glob.glob("camera_cal/calibration*.jpg") #list of chessboard image files
calib_params = camera_calibration(folder_calibration, nx, ny)

with open('calib_params.p', 'wb') as handle:
    pickle.dump(calib_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



# for a good calibration ret must be between 0.1 and 1.0
print('RMS Error of Camera calibration: {:.3f}'.format(calib_params['ret']) )
print('This number must be between 0.1 and 1.0')