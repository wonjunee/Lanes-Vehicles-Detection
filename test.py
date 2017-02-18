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
test_images = glob.glob('./test_images/*')
# Load the sample image
img_name = test_images[2]
img = cv2.imread(img_name)
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
# Perspective transform
M, Minv = perspective_transform(img)
# Unwarping corners
warped = warp_image(img, M, mtx, dist)
# For source points I'm grabbing the outer four detected corners
img_size = (img.shape[1], img.shape[0])
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
	[[(img_size[0] / 4), 0],
	[(img_size[0] / 4), img_size[1]],
	[(img_size[0] * 3 / 4), img_size[1]],
	[(img_size[0] * 3 / 4), 0]])

# Find edges from an image
binary_warped = find_edges(warped)
binary_warped[650:, :200] = 0
binary_warped[650:, 1050:] = 0
# Draw red lines on images
img_lines = draw_lines(img, src)
warped_lines = draw_lines(warped, dst)

# Visualize Warped image
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.imshow(img_lines)
ax1.set_title('Original Image')
ax2.imshow(warped_lines)
ax2.set_title('Warped Image')
ax3.imshow(binary_warped, cmap='gray')
ax3.set_title("edge")

out_img, left_fitx, right_fitx, left_fit, right_fit, ploty = find_lanes_windows(binary_warped, draw_boxes=True)
ax4.imshow(out_img)
ax4.plot(left_fitx, ploty, color='yellow')
ax4.plot(right_fitx, ploty, color='yellow')
ax4.set_xlim(0, 1280)
ax4.set_ylim(720, 0)
ax4.set_title("Windows")

result, left_fitx, right_fitx, left_fit, right_fit, ploty = find_lanes_polyfit(binary_warped, left_fit, right_fit, draw_boxes=True)
ax5.imshow(result)
ax5.plot(left_fitx, ploty, color='yellow')
ax5.plot(right_fitx, ploty, color='yellow')
ax5.set_xlim(0, 1280)
ax5.set_ylim(720, 0)
ax5.set_title("Polyfit")

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

ax6.imshow(result)
ax6.set_title('result')
plt.show()