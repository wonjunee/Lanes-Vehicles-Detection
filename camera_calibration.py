import pickle
import glob
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import *

# Define names of a pickle file
pickle_file = "wide_dist_pickle.p"
calibration_images = glob.glob('./camera_cal/*')
# Load the sample image
img_name = calibration_images[13]
img = cv2.imread(img_name)
# If a pickel file exists, then load the file
if os.path.isfile(pickle_file):
	print("A pickle file exists")
	with open(pickle_file, 'rb') as f:
		pickle_data = pickle.load(f)
		objpoints = pickle_data['objpoints']
		imgpoints = pickle_data['imgpoints']
		del pickle_data  # Free up memory
# If not found, start calibrating using example images
else:	    
	# Prepare object points
	nx = 9 # Enter the number of inside corners in x
	ny = 6 # Enter the number of inside corners in y
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((nx*ny,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.
	# Step through the list and search for chessboard corners
	# Iterate through example images to create calibration data
	for fname in calibration_images:
		# Load an image
	    img = cv2.imread(fname)
	    # Convert the image to grayscale
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    # Find the chessboard corners
	    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
	    # If found, add object points, image points
	    if ret == True:
	        print(fname)
	        # Append object poitns and corners
	        objpoints.append(objp)
	        imgpoints.append(corners)
	# Test undistortion on an image
	img_name = calibration_images[11]
	img = cv2.imread(img_name)
	img_size = (img.shape[1], img.shape[0])
	# Save the camera calibration result for later use
	dist_pickle = {}
	dist_pickle['objpoints'] = objpoints
	dist_pickle['imgpoints'] = imgpoints
	pickle.dump( dist_pickle, open( pickle_file, 'wb' ) )
# Visualize undistortion
dst = cal_undistort(img, objpoints, imgpoints)
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=15)
plt.show()