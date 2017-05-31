# Project: Search and Sample Return

[//]: # "Image References"

[image1a]: ./output/grid.PNG
[image1b]: ./output/warpedGrid.PNG
[image2a]: ./output/colorThresh.PNG
[image3]: ./output/completedThresh.PNG
[image4a]: ./output/MaptoWorld.PNG
[image4b]: ./output/rotationAndTranslation.PNG
[image4c]: ./output/rotationMatrix.PNG
[image5]: ./output/completeAutoRun.PNG
[video1]: ./output/test_mapping.MP4


##### [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Training/Calibration (Notebook Analysis)
### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

#### Perspective Transform


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

#### Color Thresholding

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
3. (Yellow) Sample/Rock extraction.  Lower RGB threshold (100, 100, 0) and upper RGB threshold (190, 190, 50)
```python 
def sample_thresh(img):
    
    # Create an array of zeros same xy size as img, but single channel
    low_yellow = np.array([100, 100, 0], dtype = "uint8")
    high_yellow = np.array([190, 190, 50], dtype = "uint8")

    sample_select = cv2.inRange(img, low_yellow, high_yellow)
    # Return the binary image
    return sample_select
```

Below are examples of Navigatable terrain extraction (left) and sample extraction (right):

![alt text][image2a]

#### Coordinate Transformations

The goal of the Coordinate Transforms is to allow you to use the rover's position, orientation and camera image to map its environment and compare against this ground truth map.

![alt text][image4a]

The environment you will be navigating with the rover in this project is roughly 200 x 200 meters and looks like the image above from a top-down view. The white areas represent the navigable terrain. You will be provided a copy of this map with the project at a resolution of 1 square meter per pixel (same as shown above). 

#### Rotation and Translation
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

#### The Resulting Process
Once we calculate pixel values in rover-centric coords and distance/angle to all pixels, apply rotation and tranlation we are provided with the following 4 images below. Upper left is a camera image. Upper right is a perspective transform. Bottom left is a combined thresholded image. The blue area is the navigable ground and the red part are the obstacles. Bottom right is a coordinate transform with and arrow indicating direction of travel of the robot.

![alt text][image3]


### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

#### 1. Define source and destination points for perspective transform

```python
dst_size = 5 
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
```

#### 2. Apply perspective transform
```python 
warped = perspect_transform(img, source, destination)
```

#### 3. Apply color threshold to identify navigable terrain/obstacles/rock samples

```python
navigable = navigate_thresh(warped) #navigable terrain color-thresholded binary image
obstacles = obstacle_thresh(warped) # obstacle color-thresholded binary image
samples = sample_thresh(warped) #sample color-thresholded binary image
```

#### 4. Convert thresholded image pixel values to rover-centric coords

```python
x_ObsPixel, y_ObsPixel  = rover_coords(obstacles)
x_SamplePixel, y_SamplePixel  = rover_coords(samples)
x_NavPixel, y_NavPixel  = rover_coords(navigable)
```

#### 5. Convert rover-centric pixel values to world coords

The coordinates in the rover frame of reference are now converted to world frame of reference using `pix_to_world()`.
```python
scale = 10 # 0.1 to 1
world_size = data.worldmap.shape[0]
    
obstacle_x_world, obstacle_y_world = pix_to_world(x_ObsPixel, y_ObsPixel, data.xpos[data.count], data.ypos[data.count], 
                                    data.yaw[data.count], world_size, scale)
sample_x_world, sample_y_world = pix_to_world(x_SamplePixel, y_SamplePixel,data.xpos[data.count], data.ypos[data.count], 
                                     data.yaw[data.count], world_size, scale)
nav_x_world, nav_y_world = pix_to_world(x_NavPixel, y_NavPixel, data.xpos[data.count], data.ypos[data.count], 
                                     data.yaw[data.count], world_size, scale)
```

#### 6. Update worldmap (to be displayed on right side of screen)

Optimizing Map Fidelity is an interesting proces.: The perspective transform is technically only valid when `roll` and `pitch angles` are near zero. If the Rovwer is slamming on the brakes or turning hard, both the `pitch` and `roll` can depart significantly from zero, and the transformed image will no longer be a valid map. It will be skewed and the resulting fidelity of what the Rover "maps" vs. the "real world" map with not coincide. Therefore, setting thresholds near zero in roll and pitch to determine which transformed images are valid for mapping before actually mapping them results in much better fidelity.
```python
#we do not want to update map if the pitch and roll are too great else our prespective
#step wont be accurate
MAXPITCH = .4 #from basepoint 0
MAXROLL = .4 # from basepoint 0

#get the minimum of both 
pitch = min(abs(data.pitch[data.count]), abs(data.pitch[data.count] -360))
roll = min(abs(data.roll[data.count]), abs(data.roll[data.count]))
if(abs(pitch) < MAXPITCH and abs(roll) < MAXROLL):
    data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    data.worldmap[sample_y_world, sample_x_world, 1] += 1
    data.worldmap[nav_y_world, nav_x_world, 2] += 1
```


#### 7. Make a mosaic image

Creating a composite image for the video.
```python
# First create a blank image (can be whatever shape you like)
output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
    # Next you can populate regions of the image with various output
    # Here I'm putting the original image in the upper left hand corner
output_image[0:img.shape[0], 0:img.shape[1]] = img

    # Let's create more images to add to the mosaic, first a warped image
warped = perspect_transform(img, source, destination)
    # Add the warped image in the upper right hand corner
output_image[0:img.shape[0], img.shape[1]:] = warped

    # Overlay worldmap with ground truth map
map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    # Flip map overlay so y-axis points upward and add to output_image 
output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)
```
#### A quick nagivation and mapping video using the functionality from the above code can be found in the output folder.

## Autonomous Navigation and Mapping : NASA Rover Challenge

### Requirements for a Passing Project Submission
The requirement for a passing submission is to map at least 40% of the environment at 60% fidelity and locate at least one of the rock samples. Each time you launch the simulator in autonomous mode there will be 6 rock samples scattered randomly about the environment and your rover will start at random orientation in the middle of the map.

### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
With regards to `perception_step()`, almost all of the explaination on how these steps were coded is provided above. 
With regards to `decision_step()`: The goal was to navigate around the map, pick up as many rocks/samples as possible and try not to get stuck.
```python
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status

        if Rover.mode == 'pickup':
            #if you see samples ahead
            if len(Rover.sample_angles) > 0:
                if Rover.near_sample and not Rover.picking_up:
                    Rover.brake = 2
                    Rover.throttle = 0
                    Rover.mode = 'stop'
                    Rover.send_pickup = True
                else:
                    Rover.steer = np.clip(np.mean(Rover.sample_angles * 180 / np.pi), -15, 15)
                    # low speed mode
                    #if not slow, slow down
                    if Rover.vel > (Rover.max_vel / 2):
                        Rover.throttle = 0
                        Rover.brake = 1
                    else:
                        #inch your way there making sure you are not above vel.
                        Rover.throttle = Rover.throttle_set
                        Rover.brake = 0

            #you just pick up the sample and are in a bad position against the wall
            else:
                Rover.mode = 'forward'
                
        
        elif Rover.mode == 'forward':
             # Check if there is rock in front of rover
            if len(Rover.sample_angles) > 0:
                Rover.steer = np.clip(np.mean(Rover.sample_angles * 180 / np.pi), -15, 15)
                Rover.mode = 'pickup'
            
            # Check the extent of navigable terrain
            elif len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set a larger throttle value at begginging
                    if Rover.vel < 0.5 and Rover.mode == 'forward':
                        Rover.throttle = Rover.throttle_set *5
                    # Set throttle value to throttle setting
                    else:
                        Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                    
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                #run along the left side by adding a bias to the mean by 10
                #Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi) + 10, -15, 15)
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
        
            
        
        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if len(Rover.sample_angles) > 0:
                Rover.steer = np.clip(np.mean(Rover.sample_angles * 180 / np.pi), -15, 15)
                Rover.mode = 'pickup'

            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'

                    
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover
```



Something to Note:
The most important aspect was Optimizing Map Fidelity for the first run. Never did it meet the fidelity specifications until I included the max pitch/roll calculations. Without these, the fidelity was usually around 40%. Not a good accuracy! After the changes to MAX pitch/roll, the perception method disgarded any perspectives that had any distortions, improving accuracy over 30%!


### 2. Launching in autonomous mode your rover can navigate and map autonomously.

The robot is capable of mapping more than 40% of the terrain at more than 60% fidelity. Sometimes the fidelity starts out below 60% but then it quickly goes up above 60%. As you can see, it mapped and grabbed 3 samples in over >50%  of the map at over > 70% fidelity.
![alt text][image5] 

A demo video can be found [YouTube](https://youtu.be/Dxi4_tZwQpI)

In addition it happens that the Rover gets stuck for a short while (a few seconds) when rotating away from certain obstacles. To counter this, the Rover is set throttle as much as it can until it can reach of speed of over 0.5 m/s. This rapid acceleration allows it to get loose and continue its exploration.

On very rare occasions where there are small random obstacles, the robot gets stuck in in awkward positions due to weird geometry of the environment/boulders and is not able to continue its exploration path. In this case the addition of a 'reverse' driving functionality in the decision_step() would be very useful.

To improve the results, most of the enhancements would be towards `decision.py`:
* Give the robot more brains as to increasing speeds in straight aways for longer periods without swaying
* Stop from re-visting areas. Have some sort of understanding of already "mapped coords".
* Making the Rover a wall crawler (since most samples are near the walls). If I made it always hug the RIGHT wall, it would never revist the same area until it reached 100% completion
* Implementing start/end position coordinates. So once the Rover has mapped and grabbed all the samples, it would return to its starting coordinates using a somewhat complex planned navigation routine.  



**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines! Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by drive_rover.py) in your writeup when you submit the project so your reviewer can reproduce your results.**


`FPS: 57`
`SIMULATOR SETTINGS: 1024x768, GRAPHICS = FANTASTIC,  VERSION: MOST UP TO DATE INSTALL`

