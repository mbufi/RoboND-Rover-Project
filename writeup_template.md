# Project: Search and Sample Return


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # "Image References"

[image1a]: ./output/grid.PNG
[image1b]: ./output/warpedGrid.PNG
[image2a]: ./output/navigableExtract.PNG
[image2b]: ./output/ObstacleExtract.PNG
[image3]: ./output/resultMapping.PNG
[image4a]: ./output/MaptoWorld.PNG
[image4b]: ./output/rotationAndTranslation.PNG
[image4c]: ./output/rotationMatrix.PNG
[video1]: ./output/test_mapping.MP4

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Training/Calibration (Notebook Analysis)
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

##### Perspective Transform


In the simulator you can toggle on a grid on the ground for calibration. This is useful for selecting four points in your "source" image and map them to four points in our "destination" image, which will be the top-down view.The grid squares on the ground in the simulator represent 1 meter square each so this mapping will also provide us with a distance estimate to everything in the ground plane in the field of view.



`perspect_transform(img, src, dst):` was defined that takes in the grid image, src points and desination points. Perspective transforms involve some complicated geometry, so the following OpenCV functions were used `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`, to aid in that process and do the calculations. The `perspect_transform(img, src, dst):` performs the Perspective Transform in the following 4 steps:
1. Define 4 source points, in this case, the 4 corners of a grid cell in the image above.
2. Define 4 destination points (must be listed in the same order as source points!).
3. Use `cv2.getPerspectiveTransform()` to get M, the transform matrix.
4. Use `cv2.warpPerspective()` to apply M and warp your image to a top-down view.

A tricky part to this is choosing the desintation points. In this case, it makes sense to choose a square set of points so that square meters in the grid are represented by square areas in the destination image. Mapping a one-square-meter grid cell in the image to a square that is 10x10 pixels, for example, implies a mapping of each pixel in the destination image to a 0.1x0.1 meter square on the ground.

```python
# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
```

Below is an example of a camera image with grid added (to the left) and this same image with perspective transform applied to it (right). In the image below, the grid cells on the ground have been mapped to pixel squares, or in other words, each pixel in this image now represents 10 cm squared in the rover environment. The result looks a bit strange, but it is now a top-down view of the world that is visible in the rover camera's field of view. Basically a map for the rover.

![alt text][image1a]  ![alt text][image1b]

##### Color Thresholding

Once the prespective transform has been applied, the next step is to create 3 new functions called `def navigate_thresh(img, rgb_thresh=(160, 160, 160)):`, `def obstacle_thresh(img,rgb_thresh=(160,160,160)):`, and `def sample_thresh(img, low_yellow_thresh=(100, 100, 0), hi_yellow_thresh=(210, 210, 55)):`. These 3 are used to extract the features out of the incoming camera input from the Rover, and segement them into binary images looking for specific thresholds. Each method takes care of providing the Rover a "Navigatable terrain Image", "Obstacles Ahead", and "Samples/special Rocks in view". The `def sample_thresh` takes lower and upper thresholds of the color yellow in RGB. The function requires that each pixel in the input image be above all three lower threshold values in RGB and below the three  upper threshold values in RGB.

Each function is called consistantly, each with a different use case:
1. Navigatable extraction. Threshold of RGB > 160 does a nice job of identifying ground pixels only
```python
def navigate_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    navigate_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    navigate_select[above_thresh] = 1
    # Return the binary image
    return navigate_select
```

2. Obstacle extraction.Threshold of RGB < 160 does a nice job of identifying obstacle pixels only
```python
def obstacle_thresh(img,rgb_thresh=(160,160,160)):
    obstacle_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    obstacle_select[below_thresh] = 1
    # Return the binary image
    return obstacle_select
```
3. (Yellow) Sample/Rock extraction.  Lower RGB threshold (100, 100, 0) and upper RGB threshold (210, 210, 55)
```python 
def sample_thresh(img, low_yellow_thresh=(100, 100, 0), hi_yellow_thresh=(210, 210, 55)):
    
    # Create an array of zeros same xy size as img, but single channel
    sample_select = np.zeros_like(img[:,:,0])
    
    # Threshold the image to get only yellow colors
    sample_mask = cv2.inRange(img, low_yellow_thresh, hi_yellow_thresh)
    
    # Index the array of zeros with the boolean array and set to 1
    sample_select[sample_mask] = 1
    # Return the binary image
    return sample_mask
```

Below are examples of Navigatable terrain extraction (left) and Obstacle extraction (right):

![alt text][image2a] ![alt text][image2b]

##### Coordinate Transformations
![alt text][image4a]

The goal of the Coordinate Transforms is to allow you to use the rover's position, orientation and camera image to map its environment and compare against this ground truth map.

The environment you will be navigating with the rover in this project is roughly 200 x 200 meters and looks like the image above from a top-down view. The white areas represent the navigable terrain. You will be provided a copy of this map with the project at a resolution of 1 square meter per pixel (same as shown above). 

###### Rotation and Translation
![alt text][image4b]

For rotation, you accomplish this by applying a rotation matrix to the rover space pixel values (xpixel,ypixel). For a roation through an angle θ, use the following:

![alt text][image4c]

The functions `rotate_pix()`is implemented as so: 

```python
# Convert yaw to radians
# Apply a rotation
yaw_rad = yaw * np.pi / 180
xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.sin(yaw_rad)
# Return the result  
```

The next step in mapping to world coordinates is to perform a translation by simply adding the x and y components of the rover's position to the x_rotated and y_rotated values calculated above.

`translate_pix()` is implemented as so:
```python
# Apply a scaling and a translation
xpix_translated = np.int_(xpos + (xpix_rot / scale))
ypix_translated = np.int_(ypos + (ypix_rot / scale))
# Return the result  
```
Note: Scale is a factor of 10 between world space pixels and rover space pixels in this case.

##### The Resulting Process

Below are 4 images showing the process. Upper left is a camera image. Upper right is a perspective transform. Bottom left is a combined thresholded image. Here the blue part is the navigable ground and the red part are the obstacles. Bottom right is a coordinate transform with the arrow indicating direction of travel of the robot.

![alt text][image3]


#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

##### 1. Define source and destination points for perspective transform

The same transform is used as above.

##### 2. Apply perspective transform

##### 3. Apply color threshold to identify navigable terrain/obstacles/rock samples

The color thresholds are applied to find the Obstacles, (Yellow) Rocks and (Navigable) Ground. The results are coordinates corresponding to the 3 categories.

##### 4. Convert thresholded image pixel values to rover-centric coords

The coordinates found above are transformed to the rover frame of reference using `rover_coords()`.

##### 5. Convert rover-centric pixel values to world coords

The coordinates in the rover frame of reference are now converted to world frame of reference using `pix_to_world()`.

##### 6. Update worldmap (to be displayed on right side of screen)

If the robot's roll or pitch are below a certain value - meaning when the robot is on a flat surface - then the obstacles, the yellow rocks and the navigable ground are added to the world map. In this case the value is set to `0.5`.

##### 7. Make a mosaic image

Creating a composite image for the video.



![alt text][video1]

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines! Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by drive_rover.py) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]

