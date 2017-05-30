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
[image2b]: ./output/obstacleExtract.PNG
[image3]: ./output/resultMapping.PNG
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
3. Use cv2.getPerspectiveTransform() to get M, the transform matrix.
4. Use cv2.warpPerspective() to apply M and warp your image to a top-down view.

A tricky part to this is choosing the desintation points. In this case, it makes sense to choose a square set of points so that square meters in the grid are represented by square areas in the destination image. Mapping a one-square-meter grid cell in the image to a square that is 10x10 pixels, for example, implies a mapping of each pixel in the destination image to a 0.1x0.1 meter square on the ground.



Below is an example of a camera image with grid added (to the left) and this same image with perspective transform applied to it (right). In the image below, the grid cells on the ground have been mapped to pixel squares, or in other words, each pixel in this image now represents 10 cm squared in the rover environment. The result looks a bit strange, but it is now a top-down view of the world that is visible in the rover camera's field of view. Basically a map for the rover.

![alt text][image1a]  ![alt text][image1b]

##### Color Thresholding

A new function is created called `color_threshold()`.  It takes lower and upper thresholds. The function requires that each pixel in the input image be above all three lower threshold values in RGB and below the three  upper threshold values in RGB.

This function is called with different thresholds for each of the 3 different cases:
1. Ground extraction. Lower RGB threshold (165,165,165) and upper RGB threshold (255,255,255)
2. Obstacle extraction. Lower RGB threshold (1,1,1) and upper RGB threshold (150,150,150)
3. (Yellow) Rock extraction.  Lower RGB threshold (130,100,0) and upper RGB threshold (200,200,70)

An example of Ground extraction (left) and Obstacle extraction (right):

![alt text][image2a] ![alt text][image2b]

##### Coordinate Transformations

The functions `rotate_pix()` and `translate_pix()` are completed respectively as follows:

```python
# Convert yaw to radians
yaw_rad = yaw * np.pi / 180
# Apply a rotation
xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.sin(yaw_rad)
```

```python
# Apply a scaling and a translation
xpix_translated = np.int_(xpos + (xpix_rot / scale))
ypix_translated = np.int_(ypos + (ypix_rot / scale))
```

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

