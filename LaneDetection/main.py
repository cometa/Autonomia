import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
#from IPython.display import HTML
from pipeline import *
import sys
import os

if __name__ == "__main__":

	try:
		vid_path = os.path.expanduser(sys.argv[1])
	except Exception as e:
		print(e, "Usage: main.py <DATA-DIR> <SOLUTION-DIR>")
		sys.exit(-1)

	video_output = 'solution.mp4'
	clip1 = VideoFileClip(vid_path)
	white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
	white_clip.write_videofile(video_output, audio=False)
