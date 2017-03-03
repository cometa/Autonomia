import numpy as np
import cv2
import pickle

class ImageProcess():
    '''
    Processing methods of original images
    '''
    def __init__(self):
        pass
    
    def gaussianBlur(self, img, k_sz=5):
        # Useful to remove salt'n pepper noise - NOT USE HERE
        img = cv2.GaussianBlur(img, (k_sz, k_sz), 0)
        return img
    
    
    def directional_gradient(self, img, direction='x', thresh=[0, 255]):
        '''
        Gradient along vertical or horizontal direction using OpenCV Sobel 
        img: Grayscale
        direction: x(horizontal) or y(vertical) for gradient direction
        thresh: apply threshold on pixel intensity of gradient image
        output is a binary image
        '''
        if direction == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        elif direction == 'y':
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        
        sobel_abs = np.absolute(sobel)  #absolute value
        scaled_sobel = np.uint8(sobel_abs * 255/np.max(sobel_abs)) #turn sobel to 8bit image 0-255 intensity range
        binary_output = np.zeros_like(sobel)
        binary_output[(scaled_sobel>= thresh[0]) & (scaled_sobel <= thresh[1]) ] = 1 #generate binary
        
        return binary_output


    def mag_gradient(self, img, thresh=[0, 255]):
        '''
        Magnitude of gradient : sqrt(gradx**2 + grady**2)
        img: RGB or Grayscale image
        thresh: apply threshold on pixel intensity of the gardient magnitude
        output is a binary image
        '''
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)  #gradient along x
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1) #gradient along y
        gradient_mag = np.sqrt( np.square(sobelx) + np.square(sobely)) # norm of gradient
        scaled_gradient_mag = np.uint8(gradient_mag * 255/np.max(gradient_mag)) #turn sobel to 8bit image 0-255 intensity range
        binary_output = np.zeros_like(gradient_mag)
        binary_output[(scaled_gradient_mag >= thresh[0]) & (scaled_gradient_mag <= thresh[1]) ] = 1 #thresholding
        
        return binary_output


    def gradient_direction(self, img, thresh=[0, 90], ksize=3):
        '''
        Direction of gradient: arctan(grady/gradx)
        img: RGB or Grayscale image
        thresh: apply threshold on gradient direction in degrees (0, 90)
        ksize: kernel size (can only be a odd number)
        output is a binary image
        '''
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize) #gradient along x
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize) #gradient along y
        gradient_dir = np.arctan2( sobely, sobelx)
        thresh = [thresh[0] * np.pi/180, thresh[1] * np.pi/180] #convert threshold from degree to radian
        binary_output = np.zeros_like(gradient_dir)
        binary_output[(gradient_dir>= thresh[0]) & (gradient_dir <= thresh[1]) ] = 1 #thresholding
        
        return binary_output

    
    def color_binary(self, img, dst_format='HLS', ch=2, ch_thresh=[0,255]):
        '''
        Color thesholding on channel ch
        img: RGB
        dst_format: destination format (HLS or HSV)
        ch_thresh: pixel intensity threshold on channel ch
        output is a binary image
        '''
        if dst_format == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            ch_binary = np.zeros_like(img[:,:, int(ch-1)])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            ch_binary = np.zeros_like(img[:,:, int(ch-1)])
            ch_binary[(img[:,:,int(ch-1)] >= ch_thresh[0]) & (img[:,:,int(ch-1)]<= ch_thresh[1])] = 1
        
        return ch_binary
    
    
    
    def image_correction(self, img, cal_params):
        '''
        correct image from camera distorsion
        img: original image RGB format
        cal_params: calibration parameters of camera (Camera Matrix and distorsion Coefficients)
        return: undistorted image
        '''

        dst = cv2.undistort(img, cal_params['cameraMatrix'], \
                            cal_params['distorsionCoeff'], None, \
                            cal_params['cameraMatrix'])
        return dst
        
    
    def convert2_rgb(self, img):
        '''
        convert image to RGB
        img: RGB image
        '''
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return rgb
        except:
            print('image cannpot be converted to RGB')
    
    
    def convert2_gray(self, img):
        '''
        convert image to gray
        img: RGB image
        '''
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return gray
        elif len(img.shape) == 2: #img channel already squashed
            return img

    
    def birdView(self, img, M):
        '''
        Transform image to birdeye view
        img: binary image
        M: transformation matrix
        return a warped image
        '''
        
        img_sz = (img.shape[1], img.shape[0])
        img_warped = cv2.warpPerspective(img, M, img_sz, flags=cv2.INTER_LINEAR)
        
        return img_warped