import cv2

# A function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,img.shape[:2][::-1],None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # For source points I'm grabbing the outer four detected corners
    src = np.float32(
	    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
	    [((img_size[0] / 6) - 10), img_size[1]],
	    [(img_size[0] * 5 / 6) + 60, img_size[1]],
	    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
	dst = np.float32(
	    [[(img_size[0] / 4), 0],
	    [(img_size[0] / 4), img_size[1]],
	    [(img_size[0] * 3 / 4), img_size[1]],
	    [(img_size[0] * 3 / 4), 0]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    # Return the resulting image and matrix
    return warped, M, Minv
