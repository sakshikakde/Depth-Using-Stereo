# Depth estimation using Stereo Cameras
## Overview
### CALIBRATION
#### 1. Feature Detection and Matching
The result are obtained using Brute-Force matcher in OpenCV. It can be seen that there are a few wrong matches obtained. These can
terribly affect the results. To filter these wrong feature pairs, we will use RANSAC in the next step.
![alt text](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/Screenshot%20from%202021-07-25%2013-07-55.png)
#### 2. Estimation of Fundamental Matrix and RANSAC
The fundamental matrix is calculated using the 8-point algorithm. If the F matrix estimation is good, then terms
x T 2 .F.x 1 should be close to 0, where x 1 and x 2 are features from image1 and image2. Using this criteria, RANSAC can
be used to filter the outliers.
![alt text](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/Screenshot%20from%202021-07-25%2013-08-01.png)
#### 3. Estimation of Essential Matrix
Since we know the camera caliberation matrix, we can use it to obtain the essential matrix.
#### 4. Estimation of Camera Pose
camera pose(R and C) can be estimated using the essential matrix E. We will be estimating
the pose for camera 2 with respect to camera 1 which is assumed to be at world origin. We will get four solutions for which we will use the Cheirality condition to xhoose the correct set.

### RECTIFICATION
Using the fundamental matrix and the feature points, we can obtain the epipolar lines for both the images. The
epipolar lines need to be parallel for further computations to obtain depth. This can be done by reprojecting image planes
onto a common plane parallel to the line between camera centers.
![alt text](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/Screenshot%20from%202021-07-25%2013-17-26.png)
### CORRESPONDENCE
For every pixel in image 1, we try to find a corresponding match along the epipolar line in image 2. We will consider a
window of a predefined size for this purpose, so this method is called block matching. Essentially, we will be taking a small
region of pixels in the left image, and searching for the closest
matching region of pixels in the right. Following methods can be used for 
block comparison:
1) Sum of Absolute Differences (SAD)
2) Sum of Squared Differences (SSD)
3) Normalized Cross-Correlation (NCC)
### Depth computation
### 1. Disparity Map
After we get the matching pixel location, the disparity can
be found bu take the absolute of the difference between the
source and matched pixel location         

![alt-text-1](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/disparity_image_gray1.png)
![alt-text-1](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/disparity_image_gray2.png)
![alt-text-1](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/disparity_image_gray3.png)
### 2. Depth Map
If we know the focal length(f) and basline(b), the depth can
be calculated.
![alt-text-1](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/depth_image1.png)
![alt-text-1](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/depth_image2.png)
![alt-text-1](https://github.com/sakshikakde/Depth-Using-Stereo/blob/main/git_images/depth_image3.png)
## How to run the code
1) Change the directory where the stereo.py file is located. Eg:      
            cd /home/sakshi/courses/ENPM673/project3_sakshi/Code  
2) Run the following command:       
            python3 stereo.py --DataPath /home/sakshi/courses/ENPM673/project3_sakshi/Data/Project\ 3/Dataset\ 3 --DataNumber 3

## Parameters:
1) DataPath: Absolute path where the images are located. Be careful about the spaces.
2) DataNumber: The dataset number. used to choose the intrinsic parameters.

