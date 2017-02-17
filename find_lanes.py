import pickle
import glob
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import *
from Line import *

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
	# If the line is detected from previous frame use polyfit
	if Line.detected:
		result, left_fitx, right_fitx, left_fit, right_fit, ploty = find_lanes_polyfit(binary_warped, left_fit, right_fit, draw_boxes=True)
	# If not then use windows search
	else:
		result, left_fitx, right_fitx, left_fit, right_fit, ploty = find_lanes_windows(binary_warped, draw_boxes=True)
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
	# Return an image with shaped area
	return result