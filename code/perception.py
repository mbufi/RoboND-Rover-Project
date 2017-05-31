import numpy as np
import cv2


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
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


# Identify pixels below the threshold
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


def sample_thresh(img):
    
    # Create an array of zeros same xy size as img, but single channel
    low_yellow = np.array([100, 100, 0], dtype = "uint8")
    high_yellow = np.array([190, 190, 50], dtype = "uint8")

    sample_select = cv2.inRange(img, low_yellow, high_yellow)
    # Return the binary image
    return sample_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.sin(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped



# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    MAXPITCH = .4 #from basepoint 0
    MAXROLL = .4 # from basepoint 0

    #get the minimum of both 
    pitch = min(abs(Rover.pitch), abs(Rover.pitch -360))
    roll = min(abs(Rover.roll), abs(Rover.roll -360))
    
    #we dont want to update if the pitch isnt good.
    image = Rover.img

    # 1) Define source and destination points for perspective transform
    
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    navigable = navigate_thresh(warped) #navigable terrain color-thresholded binary image
    obstacles = obstacle_thresh(warped) # obstacle color-thresholded binary image
    samples = sample_thresh(warped) #rock_sample color-thresholded binary image 

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)

    Rover.vision_image[:,:,0] = obstacles * 255
    Rover.vision_image[:,:,1] = samples * 255
    Rover.vision_image[:,:,2] = navigable * 255

    
    # 5) Convert map image pixel values to rover-centric coords
    x_ObsPixel, y_ObsPixel  = rover_coords(obstacles)
    x_SamplePixel, y_SamplePixel  = rover_coords(samples)
    x_NavPixel, y_NavPixel  = rover_coords(navigable)
	
    # 6) Convert rover-centric pixel values to world coordinates
    scale = 10
    world_size = Rover.worldmap.shape[0]
    xpos, ypos = Rover.pos

    obstacle_x_world, obstacle_y_world = pix_to_world(x_ObsPixel, y_ObsPixel, xpos, ypos,
                                                      Rover.yaw, world_size, scale)
    sample_x_world, sample_y_world = pix_to_world(x_SamplePixel, y_SamplePixel, xpos, ypos,
                                              Rover.yaw, world_size, scale)
    navigable_x_world, navigable_y_world = pix_to_world(x_NavPixel, y_NavPixel, xpos, ypos,
                                                        Rover.yaw, world_size, scale)
    
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)

    #we do not want to update map if the pitch and roll are too great else our prespective
    #step wont be accurate
     
    if(abs(pitch) < MAXPITCH and abs(roll) < MAXROLL):
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[sample_y_world, sample_x_world, 1] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    
    # 8) Convert rover-centric pixel positions to polar coordinates ( x/y to pixel coords)

    # Update Rover pixel distances and angles
    navigatable_distance, navigatable_angles = to_polar_coords(x_NavPixel, y_NavPixel)
    Rover.nav_dists = navigatable_distance
    Rover.nav_angles = navigatable_angles

    sample_distance, sample_angles = to_polar_coords(x_SamplePixel, y_SamplePixel)
    Rover.sample_dists = sample_distance
    Rover.sample_angles = sample_angles
    
    return Rover
