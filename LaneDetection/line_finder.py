from scipy.stats import iqr
import numpy as np
import cv2
import glob
from collections import deque

peak_thresh = 10 # if number of hot pixel in window below 50, #consider them as noise and do not attempt to get centroid

class Line():
    '''
    extract pixels associated with lanes and fit them to 2nd order polynomial function
    '''

    def __init__(self,buffer_sz):
        # was the line detected in the last iteration? 
        self.buffer_sz = buffer_sz
        #x values of hotpixels for the last n(=buffer_sz) frames
        self.allx = deque([], maxlen=self.buffer_sz)  
        #y values of hotpixels for the last n frames
        self.ally = deque([], maxlen=self.buffer_sz)
        
        #polynomial coefficients averaged over the last n iterations
        self.bestfit = {'a0':deque([], maxlen=self.buffer_sz), 
                        'a1':deque([], maxlen=self.buffer_sz), 
                        'a2':deque([], maxlen=self.buffer_sz)}
        #polynomial coefficients in real space averaged over the last n iterations
        self.bestfit_real = {'a0':deque([], maxlen=self.buffer_sz),
                                  'a1':deque([], maxlen=self.buffer_sz),
                                  'a2':deque([], maxlen=self.buffer_sz)}
        #radius of curvature of the line in m
        self.radOfCurv_tracker = deque([], maxlen=self.buffer_sz)
        
        
    def MahalanobisDist(self, x, y):
        '''
        Mahalanobis Distance for bi-variate distribution
        '''
        covariance_xy = np.cov(x,y, rowvar=0)
        inv_covariance_xy = np.linalg.inv(covariance_xy)
        xy_mean = np.mean(x),np.mean(y)
        x_diff = np.array([x_i - xy_mean[0] for x_i in x])
        y_diff = np.array([y_i - xy_mean[1] for y_i in y])
        diff_xy = np.transpose([x_diff, y_diff])
    
        md = []
        for i in range(len(diff_xy)):
            md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
        return md

    

    def MD_removeOutliers(self, x, y, MD_thresh):
        '''
        Remove pixels outliers using Mahalonobis distance
        '''
        MD = self.MahalanobisDist(x, y)
        threshold = np.mean(MD) * MD_thresh # adjust 1.5 accordingly 
        nx, ny, outliers = [], [], []
        for i in range(len(MD)):
            if MD[i] <= threshold:
                nx.append(x[i])
                ny.append(y[i])
            else:
                outliers.append(i) # position of removed pair
        return (nx, ny)


    
    def polynomial_fit(self, data):
        '''
        Perform 2nd order polynomial fit: a0 + a1 x + a2 x**2
        data: dictionary with x and y values {'x':[], 'y':[]}
        '''
        a2, a1, a0 = np.polyfit(data['x'], data['y'], 2)
        return {'a0': a0, 'a1': a1, 'a2': a2}


    
    def find_starter_centroids(self, image, x0, peak_thresh):
        '''
        Find starter centroids using histogram
        peak_thresh: If peak intensity is below a threshold use histogram on the full height of the image
        returns x-position of centroid and peak intensity
        '''
        #Define window
        window = {'x0': x0, 'y0': image.shape[0], 'width':image.shape[1]/2, 'height':image.shape[0]/2}
        
        #get centroid
        centroid, peak_intensity, _ = self.find_centroid(image, peak_thresh, window)
        #if peak intensity smaller than threshold, change window height to full image height
        if (peak_intensity < peak_thresh):
            window['height']=image.shape[0]
            centroid, peak_intensity, _ = self.find_centroid(image, peak_thresh, window)
        return {'centroid': centroid, 'intensity': peak_intensity}

    
    
    def find_centroid(self, image, peak_thresh, window):
        '''
        find centroid in a window using histogram of hotpixels
        img: binary image
        window with specs {'x0', 'y0', 'width', 'height'}
            (x0,y0) coordinates of bottom-left corner of window
        return  x-position of centroid, peak intensity and hotpixels_cnt in window 
        '''
        #crop image to window dimension
        mask_window = image[ int( round( window['y0'] - window['height'])) : int( round(window['y0'])), 
                              int( round(window['x0'])) :int(round(window['x0']+ window['width']))]
        histogram = np.sum(mask_window, axis=0)
        centroid = np.argmax(histogram)
        hotpixels_cnt = np.sum(histogram)
        peak_intensity = histogram[centroid]
        if peak_intensity <= peak_thresh:
            #centroid reading is likely unreliable - take center of box as centroid
            #global position of centroid in image
            centroid = int( round(window['x0'] + window['width']/2) )
            peak_intensity=0
        else:
            #global position of centroid in image
            centroid = int( round(centroid + window['x0']) )
        
        return (centroid, peak_intensity, hotpixels_cnt)   
  

 
    def run_sliding_window(self, image, centroid_starter, sliding_window_specs):
        '''
        Run sliding window from bottom to top of the image and return indexes of the hotpixels associated with lane
        image: binary image
        centroid_starter: centroid starting location sliding window
        sliding_window_specs: ['width', 'n_steps']
            width of sliding window
            number of steps of sliding window along vertical axis
        returns {'x':[], 'y':[]} coordinates of all hotpixels detected by sliding window
                coordinate of all centroids recorded but not used yet! 
                
        '''
        #assert image.shape[0]%n_steps==0, 'number of steps must be a factor of the image height'
        
        #Initialize sliding window
        window = {'x0': centroid_starter - int(sliding_window_specs['width']/2), 'y0': image.shape[0], 
                  'width': sliding_window_specs['width'], 
                  'height': int(round(image.shape[0]/sliding_window_specs['n_steps']))}

        #Initialize log to store coordinates of hotpixels and log to store centroids coordinates at each step
        hotpixels_log = { 'x': [], 'y':[]}
        centroids_log = []
        
        for step in range(sliding_window_specs['n_steps']):
            #Limit lateral position of window: must remains within image width
            if (window['x0'] < 0): window['x0'] = 0   
            if (window['x0'] + sliding_window_specs['width']) > image.shape[1]: 
                window['x0'] = image.shape[1] - sliding_window_specs['width']
            
            centroid, peak_intensity, hotpixels_cnt = self.find_centroid(image, peak_thresh, window)
            
            #if >60% of window area is filled by hotpixels, increase window width
            if hotpixels_cnt/(window['width']*window['height']) > 0.6:
                window['width']= window['width']*2
                window['x0'] = int(round(window['x0'] - window['width']/2))
                #Make sure window remains within image width
                if (window['x0'] < 0): window['x0'] = 0   
                if (window['x0'] + window['width']) > image.shape[1]: 
                    window['x0'] = image.shape[1] - window['width']
                centroid, peak_intensity, hotpixels_cnt = self.find_centroid(image, peak_thresh, window)

            # Create a copy of image where all pixels outside window are turned off (=0)
            mask_window = np.zeros_like(image)
            mask_window[ window['y0']- window['height'] : window['y0'],
                                window['x0']:window['x0']+window['width']] \
                    = image[ window['y0']- window['height'] : window['y0'],
                                window['x0']:window['x0']+window['width']]
            
            #Get coordinates of hot pixels in window
            hotpixels = np.nonzero( mask_window )
            hotpixels_log['x'].extend(hotpixels[0].tolist())
            hotpixels_log['y'].extend(hotpixels[1].tolist())
            #update record of centroids
            centroids_log.append(centroid)
           
            #set next position of window and use standard sliding window width
            window['width'] = sliding_window_specs['width']
            window['x0'] = int( round(centroid - window['width']/2) )
            window['y0'] = window['y0'] - window['height']
        
        return hotpixels_log

    
    
    def predict_line(self, x0, xmax, coeffs):
        '''
        Predict road line using polyfit coefficients 
        x values are in range (x0, xmax)
        polyfit coeffs: {'a2': , 'a1': , 'a2': }
        returns array of [x, y] predicted points, x along image vertical / y along image horizontal direction
        '''
        x_pts = np.linspace(x0, xmax-1, num=xmax) #x coordinates are along the vertical axis of the image
        #predict y coordinates along the horizontal axis
        pred = coeffs['a2']*x_pts**2 + coeffs['a1']*x_pts + coeffs['a0']
        
        return np.column_stack((x_pts,pred))
 

    
    def update_tracker(self, tracker, new_value):
        '''
        update tracker (self.bestfit or self.bestfit_real or radOfCurv or hotpixels) with new coeffs
        new_coeffs is of the form {'a2': val2, 'a1': val1, 'a0': val0}
        tracker is of the form {'a2': [val2,...], 'a1': [val1,...], 'a0': [val0,...]}
        update tracker of radius of curvature
        update allx and ally with hotpixels coordinates from last sliding window 
        '''
        if tracker == 'bestfit':
            self.bestfit['a0'].append(new_value['a0'])
            self.bestfit['a1'].append(new_value['a1'])
            self.bestfit['a2'].append(new_value['a2'])
        elif tracker == 'bestfit_real':
            self.bestfit_real['a0'].append(new_value['a0'])
            self.bestfit_real['a1'].append(new_value['a1'])
            self.bestfit_real['a2'].append(new_value['a2'])
        elif tracker == 'radOfCurvature':
            self.radOfCurv_tracker.append(new_value)
        elif tracker == 'hotpixels':
            self.allx.append(new_value['x'])
            self.ally.append(new_value['y'])
    
    
    
    def compute_radOfCurvature(self, coeffs, pt):
        '''
        compute radius of curvature in meter or pixels
        polyfit coeffs is of the form {'a2': val2, 'a1': val1, 'a0': val0}
        pt is the x coordinate (position along the vertical axis ) where to evaluate the radius of curvature
        '''
        return ((1 + (2*coeffs['a2']*pt + coeffs['a1'])**2)**1.5) / np.absolute(2*coeffs['a2'])
                
    

    def intercept_is_outlier(self, data, elt):
        '''
        Determine if intercept 'elt' is an outlier when compared to previous 'intercepts' in data
        returns True if elt is an outlier
        '''
        outlier_flag = False
        #evaluate if elt is an outlier when data has enough datapoints
        if len(data) == self.buffer_sz:
            p = np.min(data)-50
            q = np.max(data)+50
            
            if elt < q and elt > p:
                return False
            else:
                return True
    
    
    def is_outlier(self, data, elt):
        '''
        Determine if 'elt' is an outlier when compared to datapoints in data
        Use IQR scheme
        returns True if elt is an outlier
        NOT USED
        '''
        outlier_flag = False
        #evaluate if elt is an outlier when data has enough datapoints
        if len(data) == self.buffer_sz:
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3-q1
            if elt < (q3 + 1.5*iqr) and elt > q1 - 1.5*iqr: 
                return False
            else: 
                return True
            

        
    def approve_line(self, coeffs, xmax):
        '''
        Approve if detected hotpixels are from a real line road
        Scheme: if intercept of bestfitat 0 and xmax(bottom of image) agrees with previous frames, then flag True
        output: flag
        '''
        flag_line = True
            
        #check if intercepts at top of image is an outlier
        if self.intercept_is_outlier(self.bestfit['a0'], coeffs['a0']):
            flag_line = False

        #check if intercepts at bottom of image is an outlier
        #Calculate intercept at bottom of image for n previous frames 
        intercepts_bottom = np.array(self.bestfit['a2']) * xmax**2 + np.array(self.bestfit['a1']) * xmax \
                        + np.array(self.bestfit['a0'])
        #current frame 
        this_intercept_bottom = coeffs['a2']* xmax**2 + coeffs['a1']* xmax + coeffs['a0']
           
        if self.intercept_is_outlier(intercepts_bottom, this_intercept_bottom):
            flag_line = False
        
        #check if radius of curvature (px unit) consistent with previous curvature:
        #this_curvature_rad = self.compute_radOfCurvature(coeffs, xmax, xm_per_pix=None, ym_per_pix=None)    
        #if self.is_outlier(self.radius_of_curvature_tracker, this_curvature_rad):
        #    flag_tracker = False
        
        #get distance image center to lane line
        #if np.abs(dist_2lane) > max_dist:
        #    flag_lane = False
        #    print('False because of lane distance')
        
        #if self.is_outlier(self.bestfit['a1'], coeffs['a1']) or self.is_outlier(self.bestfit['a0'], coeffs['a0']): 
        #    flag_lane = False
        #    print('False because of outlier')
        #print('Approve Lane:', flag_lane)
        #print('**************')
        return flag_line
        
    
    def mva_smoothing(self, tracker, weighted=False):
        '''
        Moving average smoothing of polyfit coefficients 
        weighted: True, use weighted average 
                (1a + 1/2b + 1/3c...)/(1+1/2+1/3...) where a is the most recent frame, b 2nd most recent, etc...
                False: use mean
        '''
        if weighted:
            if tracker == 'coeffs':
                smooth_tracker = {'a2':0, 'a1':0, 'a0': 0}
                a2, a1, a0, denominator = 0, 0, 0, 0
                #higher weight for latest coefficients frames
                
                for i in range(len(self.bestfit['a2'])):
                    a2 = a2 + self.bestfit['a2'][i]/abs(len(self.bestfit['a2']) - i)
                    a1 = a1 + self.bestfit['a1'][i]/abs(len(self.bestfit['a2']) - i)
                    a0 = a0 + self.bestfit['a0'][i]/abs(len(self.bestfit['a2']) - i)
                    denominator = denominator + 1/abs(len(self.bestfit['a2']) - i)
                smooth_tracker['a2'] = a2/denominator
                smooth_tracker['a1'] = a1/denominator
                smooth_tracker['a0'] = a0/denominator
                return smooth_tracker
            elif tracker == 'radCurv':
                smooth_val, denominator = 0, 0
                for i in range(len(self.radOfCurv_tracker)):
                    smooth_val = smooth_val + self.radOfCurv_tracker[i]/abs(len(self.radOfCurv_tracker) - i)
                    denominator = denominator + 1/abs(len(self.radOfCurv_tracker) - i)
                return smooth_val/denominator  
        else:
            if tracker == 'coeffs':
                smooth_coeffs = {'a2':0, 'a1':0, 'a0': 0}
                smooth_coeffs['a2'] = np.mean(self.bestfit['a2'])
                smooth_coeffs['a1'] = np.mean(self.bestfit['a1'])
                smooth_coeffs['a0'] = np.mean(self.bestfit['a0'])
                return smooth_coeffs
            elif tracker == 'radCurv':
                return np.mean(self.radOfCurv_tracker)