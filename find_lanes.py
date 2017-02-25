import pickle
import glob
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import *
from Line import *

### Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Weight Parameter
w = 3.

# Define names of a pickle file
pickle_file = "wide_dist_pickle.p"
# If a pickel file exists, then load the file
if os.path.isfile(pickle_file):
	with open(pickle_file, 'rb') as f:
		pickle_data = pickle.load(f)
		mtx = pickle_data['mtx']
		dist = pickle_data['dist']
		objpoints = pickle_data['objpoints']
		imgpoints = pickle_data['imgpoints']
		del pickle_data  # Free up memory
	print("A pickle file loaded")
else:
	print("A pickle file does not exist")



def pipeline(img):
	# Perspective transform
	M, Minv = perspective_transform(img)
	# Unwarping corners
	warped = warp_image(img, M, mtx, dist)
	# Find edges from an image
	binary_warped = find_edges(warped)
	# Mask some areas
	binary_warped[650:, :200] = 0
	binary_warped[650:, 1050:] = 0
	# use windows search
	result, left_fitx, right_fitx, left_fit, right_fit, ploty = find_lanes_windows(binary_warped, draw_boxes=True)

	left_fitx_raw = copy.copy(left_fitx)
	right_fitx_raw = copy.copy(right_fitx)
	
	# Sanity check for the lanes
	left_fitx  = sanity_check(left_lane, ploty, left_fitx, left_fit)
	right_fitx = sanity_check(right_lane, ploty, right_fitx, right_fit)
	# Create an empty numpy array and draw image and shade area between lanes
	warp_zero  = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	newwarp = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
	
	# Convert grayscale into color scales
	warped_stacked = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)
    # draw lines on image
	# Left Lane
	warped_stacked_with_lines = copy.copy(warped_stacked)
	# draw polynomial lines on images
	draw_lines(pts_left, warped_stacked_with_lines)
	draw_lines(pts_right, warped_stacked_with_lines)
	# Multiple windows for diagnostics
	combined = np.zeros((720+360, 1920, 3), dtype=np.uint8)
	# main window
	combined[:720, 0:1280] = newwarp
	# right 1
	combined[:360, 1280:] = cv2.resize(warped, (640,360))
	# right 2
	combined[360:720, 1280:] = cv2.resize(result, (640,360))
	# bottom 1
	combined[720:, 0:640] = cv2.resize(warped_stacked, (640,360))
	# bottom 2
	combined[720:, 640:1280] = cv2.resize(warped_stacked_no_sanity, (640,360))
	# bottom 3
	combined[720:, 1280:] = cv2.resize(warped_stacked_with_lines, (640,360))

	# cv2.putText(combined,"Masked Image",(1500,100), font, 1,(255,255,255),2)
	# cv2.putText(combined,"Warped Image",(1500,450), font, 1,(255,255,255),2)    

	# Return an image with shaped area
	return combined

# Define the version from the command line
parser = argparse.ArgumentParser(description='Create A Movie Using A Pipeline')
parser.add_argument('movie', type=str, help='A movie file ex. project_video.mp4')
args = parser.parse_args()
movie = args.movie

# Set up lines for left and right
left_lane = Line()
right_lane = Line()
white_output = 'white.mp4'
clip1 = VideoFileClip(movie)
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)