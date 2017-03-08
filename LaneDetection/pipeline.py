import numpy as np
import cv2
from skimage import morphology
from line_finder import *
from img_process import *
import calibration

def perspective_transform(src_pts, dst_pts):
    '''
    perspective transform
    args: source and destination points
    return M and Minv
    '''
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    return {'M': M, 'Minv':Minv}


#To confirm that your detected lane lines are real, you might consider:
#Checking that they have similar curvature
#Checking that they are separated by approximately the right distance horizontally
#Checking that they are roughly parallel

#load camera calibration
with open('calib_params.p', 'rb') as handle:
    cal_params = pickle.load(handle)


camera_calib = cal_params
src_pts = np.float32([[240, 720], [575, 470], [735, 470], [1200, 720]])
dst_pts = np.float32([[240, 720], [240, 0], [1200, 0], [1200, 720]])
transform_matrix = perspective_transform(src_pts, dst_pts)
gradx_thresh=[25, 255]
ch_thresh=[50, 255]
bottom_crop = -40 #front-end car
sliding_window_specs = {'width': 120, 'n_steps': 10} #number of steps vertical steps of sliding window

buffer_sz = 9
ym_per_pix = 12/450. # meters per pixel in y dimension
xm_per_pix = 3.7/911 # meters per pixel in x dimensio

min_sz = 50
apply_MDOutlier = False
MD_thresh = 1.8

lineLeft = Line(buffer_sz=buffer_sz)
lineRight = Line(buffer_sz=buffer_sz)
alpha = None

def pipeline(image):
    '''
    Image processing to highlight lanes
    '''

    # Image processing pipeline
    process = ImageProcess()
    img_sz = (image.shape[1], image.shape[0])
    pt_curvature = image.shape[0]
    original = image.copy()

    image = process.image_correction(image, camera_calib)
        
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    gradx = process.directional_gradient(gray, direction='x', thresh=gradx_thresh )
    
    ch3_hls_binary = process.color_binary(image, dst_format='HLS', ch=3, ch_thresh=ch_thresh)
        
    combined_output = np.zeros_like(gradx)
    combined_output[((gradx == 1) | (ch3_hls_binary == 1) )] = 1

    #apply ROI mask
    mask = np.zeros_like(combined_output)
    vertices = np.array([[(100, 720), (545, 470), (755, 470), (1290, 720)]], dtype=np.int32)
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, 1)
    masked_image = cv2.bitwise_and(combined_output, mask)
    
    #Removing small aggregate of hotpixels
    cleaned = morphology.remove_small_objects(masked_image.astype('bool'), min_size=min_sz, connectivity=2)
    
    warped_img = process.birdView(cleaned*1.0, transform_matrix['M'])
    
    #adjust image height (remove front-end of car)
    warped_img = warped_img[0:bottom_crop, :]

    
    #Lane Detection Pipeline
    #Find starter
    centroid_starter_right = lineRight.find_starter_centroids(warped_img, x0=warped_img.shape[1]/2, 
                                                               peak_thresh=peak_thresh)
    
    centroid_starter_left = lineLeft.find_starter_centroids(warped_img, x0=0, peak_thresh=peak_thresh)

    # keep record left/right lane hotpixels coordinates x and y
    log_lineRight = lineRight.run_sliding_window(warped_img, centroid_starter_right['centroid'],
                                                  sliding_window_specs)
    log_lineLeft = lineLeft.run_sliding_window(warped_img, centroid_starter_left['centroid'],
                                                  sliding_window_specs)
    
    if apply_MDOutlier:
        #Remove bi-variate outliers using Mahalanobis Distance
        log_lineRight['x'], log_lineRight['y'] = \
                    lineRight.MD_removeOutliers(log_lineRight['x'], log_lineRight['y'], MD_thresh)

        log_lineLeft['x'], log_lineLeft['y'] = \
                       lineLeft.MD_removeOutliers(log_lineLeft['x'], log_lineLeft['y'], MD_thresh)
    
    
    #add this frame' hotpixels to allx and ally tracker
    lineRight.update_tracker('hotpixels', log_lineRight)
    lineLeft.update_tracker('hotpixels', log_lineLeft)
    

    # use all hotpixels accumulated in allx and ally from the last n frames
    # allx is of the form [[hotpixels frame1], [hotpixels_frame2], ....]
    multiframe_r = {'x': [val for sublist in lineRight.allx for val in sublist],
                                            'y': [val for sublist in lineRight.ally for val in sublist] }
    
    multiframe_l = {'x': [val for sublist in lineLeft.allx for val in sublist],
                                            'y': [val for sublist in lineLeft.ally for val in sublist] }

    if len(multiframe_r['x']) == 0 or len(multiframe_l['x']) == 0:
        fit_lineRight = {'a2': lineRight.bestfit['a2'][-1], 'a1': lineRight.bestfit['a1'][-1],
                        'a0': lineRight.bestfit['a0'][-1]}
        #use radius of curvature of previous frame
        radOfCurv_r = lineRight.radOfCurv_tracker[-1]
        # use coeffs of the previous frame 
        fit_lineLeft = {'a2': lineLeft.bestfit['a2'][-1], 
                         'a1': lineLeft.bestfit['a1'][-1],
                         'a0': lineLeft.bestfit['a0'][-1]}
        #use radius of curvature of previous frame
        radOfCurv_l = lineLeft.radOfCurv_tracker[-1]

    else:
        #fit to polynomial in pixel space: right line
        fit_lineRight = lineRight.polynomial_fit(multiframe_r)
        #fit to polynomial in real space: right line
        fit_lineRight_real = lineRight.polynomial_fit({'x': [i*ym_per_pix for i in multiframe_r['x']], 'y': [i*xm_per_pix for i in multiframe_r['y']]})
        # fit to polynomial in pixel space: left line
        fit_lineLeft = lineLeft.polynomial_fit(multiframe_l)
        #fit to polynomial in real space: left line
        fit_lineLeft_real = lineLeft.polynomial_fit({'x': [i*ym_per_pix for i in multiframe_l['x']], 'y': [i*xm_per_pix for i in multiframe_l['y']]})
    
    
        # check approval of fitted right line
        if lineRight.approve_line(fit_lineRight, xmax=image.shape[0]):
            # update trackers
            lineRight.update_tracker('bestfit', fit_lineRight)
            lineRight.update_tracker('bestfit_real', fit_lineRight_real)
            radOfCurv_r = lineRight.compute_radOfCurvature(fit_lineRight_real, pt_curvature*ym_per_pix)
            lineRight.update_tracker('radOfCurvature', radOfCurv_r)
        else:
            # use coeffs of the previous frame 
            fit_lineRight = {'a2': lineRight.bestfit['a2'][-1], 'a1': lineRight.bestfit['a1'][-1],
                        'a0': lineRight.bestfit['a0'][-1]}
            #use radius of curvature of previous frame
            radOfCurv_r = lineRight.radOfCurv_tracker[-1]
        
        # check approval of fitted left line
        if lineLeft.approve_line(fit_lineLeft, xmax=image.shape[0]):
            #update trackers
            lineLeft.update_tracker('bestfit', fit_lineLeft)
            lineLeft.update_tracker('bestfit_real', fit_lineLeft_real)
            radOfCurv_l = lineLeft.compute_radOfCurvature(fit_lineLeft_real, pt_curvature*ym_per_pix)
            lineLeft.update_tracker('radOfCurvature', radOfCurv_l)
        else:
            # use coeffs of the previous frame 
            fit_lineLeft = {'a2': lineLeft.bestfit['a2'][-1], 'a1': lineLeft.bestfit['a1'][-1], 'a0': lineLeft.bestfit['a0'][-1]}
            #use radius of curvature of previous frame
            radOfCurv_l = lineLeft.radOfCurv_tracker[-1]
        
    
    #display lane and best polynomial fits
    var_pts = np.linspace(0, image.shape[0]-1, num=image.shape[0])
    
    
    #smoothing fitcoeffs and radis of Curvature
    smoothfit_lineLeft = lineLeft.mva_smoothing('coeffs', weighted=True)
    radCurv_smooth_lineLeft = lineLeft.mva_smoothing('radCurv', weighted=True)
    smoothfit_lineRight = lineRight.mva_smoothing('coeffs', weighted=True)
    radCurv_smooth_lineRight = lineRight.mva_smoothing('radCurv', weighted=True)
    #predicted smoothed lane left lane
    pred_smooth_lineLeft = lineLeft.predict_line(0, image.shape[0], smoothfit_lineLeft)
    pred_smooth_lineRight = lineRight.predict_line(0, image.shape[0], smoothfit_lineRight)
    
    ###############
    # estimate offsetclose to driver (bottom of image)
    center_of_lane = (pred_smooth_lineLeft[:,1][-1] +  pred_smooth_lineRight[:,1][-1])/2
    offset = (image.shape[1]/2 - center_of_lane ) * xm_per_pix
    side_pos = 'right'
    if offset < 0:
        side_pos = 'left'


    # Create an image to draw the lines on
    warp_zero = np.zeros_like(gray).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fitx = smoothfit_lineLeft['a2']*var_pts**2 + smoothfit_lineLeft['a1']*var_pts + smoothfit_lineLeft['a0']
    right_fitx = smoothfit_lineRight['a2']*var_pts**2 + smoothfit_lineRight['a1']*var_pts + smoothfit_lineRight['a0']
    pts_left = np.array([np.transpose(np.vstack([left_fitx, var_pts]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, var_pts])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.putText(color_warp, '|', (int(image.shape[1]/2), image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    #middle of the lane
    cv2.putText(color_warp, '|', (int(center_of_lane), image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 8)
    newwarp = cv2.warpPerspective(color_warp, transform_matrix['Minv'], (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    ##############
    # Radius of Curvature
    # Determine polynomial parameters in real space
    ####fit_lineRight = .polynomial_fit(multiframe_r)
    
    # Combine the result with the original image
    
    average_radCurv = (radCurv_smooth_lineLeft + radCurv_smooth_lineRight)/2
    cv2.putText(result, 'Vehicle is '+ str(round(offset, 3)) +'m ' +side_pos +' of center',
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
    cv2.putText(result, 'Radius of curvature :'+ str(round(average_radCurv))+ 'm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)


    return result.astype('uint8')