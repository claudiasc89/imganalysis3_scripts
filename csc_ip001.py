#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:58:10 2023

@author: claudiasalatcanela
"""

"""
This script performs Z-projections on multi-stack images, automatically identifying the best-focused plane and computing the desired projection 
(average, maximum, or best focus).

The script is designed to handle 3D or 4D images (time, z, x, y) in TIFF or ND2 formats and offers tailored strategies for finding the best 
focus in either brightfield (BF) or fluorescent (F) channels.

### **Functionality:**

- **Focus Detection**: Identifies the best-focused plane within the stack, adapting the focus strategy based on the type of imaging channel 
    (brightfield or fluorescent)(coming in the next version, currently it only works for fluorescent channels)

- **Projection Types**: Supports three types of projections:
  - **Mean Projection**: Averages pixel values across selected planes.
  - **Max Projection**: Takes the maximum pixel value across selected planes.
  - **Best Focus Plane**: Selects the plane with the sharpest focus.

- **Customizable Depth**: Allows the user to specify the number of stacks above and below the best-focused plane to include in the projection.


### **Input:**

- A folder containing 3D or 4D images in TIFF or ND2 format.


### **Parameters:**

- **Projection Type**: Specify the desired projection type ('mean', 'max', 'best').
- **Input Type**: Indicate the channel type ('BF' for brightfield, 'F' for fluorescent).
- **Stack Depth**: Define the number of stacks to include above and below the best-focused plane.


### **Output:**

- A log file detailing the selected stacks for each image.
- The projected image(s) based on the specified projection type.


### **File Naming Requirements:**

- Images should be named following this pattern: `sampleID_channelname_suffixes`.
  - Example: `smp00_WL508_suffix_suffix`.
- The script considers only the first two blocks of the filename (`sampleID_channelname`), with additional suffixes ignored.


### **Requirements:**
- The script requires images to be named consistently, as described above, to correctly process and project the stacks.
- The script requires that you know in which order how the dimensions are defined in your image, you can quickly 
check using img.shape function in the terminal

### **Limitations:**
- All the images in the folder must be of the same 'type'. The folder can't not contain images with different dimensions arrangements.
  Number of time-points or z-stacks and xy dimensions can differ.


"""


#library imports
import nd2
import os
import numpy as np
from tifffile import imsave, imread


# User-configurable parameters
# ============================
# Define the folder path containing the images
folder= '/Users/claudiasalatcanela/Desktop/z_proj_img/timelapse'

# Define the type of projection to be applied: 'mean', 'max', or 'best'
proj_type='mean'

# Define the number of stacks to include above and below the best-focused plan
stack_step= 1 # it means it will get -n, best, +n 

# Define the channel type: BF for brightfield F for Fluoresencent images 
ch_type= 'F' #currently it only works for fluorescent images

# Define the channel name
ch_name= ['WL508'] # it can be a list to create projections of several fluorescent channels at a time

# Define the dimensions arrangement
dimensions_type= ('t','z','x','y') #t for time, z for z_stack, x and y dimensions


# End of user-configurable parameters
# ===================================

# Locally-defined functions
# ============================

def filenaming(folder,file):
    '''
    Generates file paths and names based on the given folder, file, and projection type.

    Parameters
    ----------
    folder : str
        The directory path where the file is located.
    file : str
        The name of the file, including its extension.
    proj_type : str
        The type of projection to include in the saved file name.

    Returns
    -------
    tuple
        A tuple containing the following:
        - filepath (str): The full path to the file.
        - filename (str): The base name of the file.
        - savename (str): The name to use when saving the file, including the projection type.
        - basename (str): The base name of the file without the extension.
    '''
    
    filepath = os.path.join(folder, file)
    filename = os.path.basename(filepath)
    savename = str(filename[:filename.index(".")])+'_'+str(proj_type)+'.tif'
    basename = str(filename[:filename.index(".")])
    #ovname=str(filename[:filename.index("_")])+'_coloc'+'.jpg'
    names=(filepath,filename,savename,basename)
    return names

def focused_z(image, dimensions_indices):
    '''
    Determines the most focused z-slice in a multi-dimensional image array by calculating the standard deviation
    of each slice along the z-axis (depth) and identifying the slice with the highest standard deviation.

    Parameters
    ----------
    image : numpy.ndarray
        A 3D array representing an image stack. 

    dimensions_indices : dict
        A dictionary mapping dimension names (e.g., 'x', 'y', 'z') to their corresponding axis indices in the `image` array.
        This allows the function to correctly identify which axes correspond to the spatial dimensions (X and Y) 
        and the depth dimension (Z).

    Returns
    -------
    tuple
        A tuple containing:
        - sd (numpy.ndarray): An array of standard deviation values for each z-slice, with the same length as the number 
          of slices along the z-axis.
        - maxsd (int): The index of the z-slice with the highest standard deviation, indicating the most focused slice.
    '''
    axis=(dimensions_indices['x'],dimensions_indices['y'])
    sd=image.std(axis=axis) #by axis we are telling it to flatten the first and second dimensions, so Y and X and get the sd of all the values
    maxsd=sd.argmax() 
    return sd, maxsd
'''
def focused_z_v2(image):
    'This function determines the most focused z slice within a multi-stack brightfield image'
    'to do so it calculates the variance between the sd deviation of consecutive images'
    sd=image.std(axis=(1,2)) #by axis we are telling it to flatten the first and second dimensions, so Y and X and get the sd of all the values
    diff_sd=np.diff(sd, prepend=sd[0]) #to calculate differences between consecutive sd values, we used prepend to add a 0 at the first row and mantain the same lenght than sd
    #I create a dictionary to store this values
    keys=['std','diff_std']
    values=[sd, diff_sd]
    sd_dict=dict(zip(keys,values))#create a new dictionary to store the calculations
    sd_df=pd.DataFrame(sd_dict)
    minsddiff=diff_sd.argmin() #we have to substract one to correclty guess the z-stack where the image is best focused
    return sd_df, minsddiff
'''

def import_image(filepath):
    '''
    Imports an image from a file, supporting TIFF and ND2 formats. 

    Depending on the file extension, the function selects the appropriate method to read the image.

    Parameters
    ----------
    filepath : str
        The full path to the image file that needs to be imported. 
        
    Returns
    -------
    img : numpy.ndarray
        The imported image as a multi-dimensional NumPy array.

    Raises
    ------
    ValueError
        If the file format is not supported (neither TIFF nor ND2).
    '''
    # Determine file extension and import image accordingly
    if filename.endswith('.tif'):
        img = imread(filepath)  # Import TIFF file
    elif filename.endswith('.nd2'):
        img = nd2.imread(filepath) # Import ND2 file
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return img

def projection_3D (img, dimensions_indices):
    '''
    Performs a specified projection on a 3D image stack.

    This function supports "best", "mean", and "max" projection types. For the "best" type, it selects the z-slice with the highest 
    standard deviation. For "mean" and "max" types, it calculates a projection based on a defined stack depth around the most focused 
    z-slice.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D image stack to be projected.

    Returns
    -------
    None
        The function saves the projected image and logs the projection details in a text file.
    '''    
    global proj_type
    global maxsd
    global filename
    global stack_step
    global dimensions
    global dimensions_map
    
    # Get variables from dimensions_map
    z = dimensions_map['z']
    
    # Get the two first blocks of filename 
    parts = filename.split('_')
    output_name = '_'.join(parts[:2])
    
    # Create a log file to store the information about the z-stack used
    log_file = os.path.join(folder,'projections', f"{output_name}_{proj_type}_log.txt")
    
    if proj_type == 'best':
        img_proj = img[maxsd,:,:]
        
        # Log the z-stack used
        with open(log_file, 'a') as f:
            log_message = f"Image: {filename}, z-stack used: {maxsd}\n"
            f.write(log_message)
    
    elif proj_type == 'mean' or proj_type == 'max':
        # Calculate stack depth
        stack_depth = stack_step*2 + 1 # we duplicate the step to include stacks below and above and add 1 to include the central stack
        
        # Check that stack_depth is smaller or equal than image dimensions
        if stack_depth > z :
            raise ValueError(f"For image {filename} stack depth should be smaller or equal than image dimensions")
        
        else:
            # To calculate the first and last stack included in the projection according to the best focused
            start_z = maxsd-stack_step
            stop_z = maxsd+stack_step+1 # add 1 to make sure that the stop_z is included
            
            # it might be than the best focused is not central so there are not enough z-stacks to accomodate our will
            # to check if best focused is closer to the first stack
            if start_z<0:
                # define the z_range that will be used for the projection, in this case we use the first z-stacks
                z_range = range(0,stack_depth)
            # to check if best focused is closer to the last stack
            elif stop_z > z:
                # in this case we use the last z-stacks
                z_range = range(z-stack_depth,z)
            else:
                # in this case the best focused plane is central so we simply sue start_z and stop_z
                z_range = range(start_z,stop_z) #we add one so it the stop stack is included
         
            # Log the z-stack used (we use different approaches for time-lapses)
            if 't' in dimensions_map:
                with open(log_file, 'a') as f:
                 log_message = f"Image: {filename},timepoint {t}, z-stack range used: {z_range}\n"
                 f.write(log_message)
            else:
                with open(log_file, 'a') as f:
                 log_message = f"Image: {filename}, z-stack range used: {z_range}\n"
                 f.write(log_message)
    
        # to slice the image
        img_z = img.take(z_range, axis=dimensions_indices['z']) # find the axis where z is stored
        
        # Perform the appropiate projection
        
        if proj_type =='mean':
            # use numpy to perform the mean projeaction
            img_proj=np.mean(img_z,axis=dimensions_indices['z'], dtype=np.uint16) #I perform mean projection
        elif proj_type =='max':
            # use numpy to perform the mean projeaction
            img_proj=np.max(img_z,axis=dimensions_indices['z']) #I perform mean projection
            # Max does not require dtype because it returns the maximum value directly, maintaining the same data type as the input.
        
    return img_proj

def get_dimensions(image):
    '''
    This function extracts the dimensions of an image based on predefined dimensions types and arrangements.
    
    Parameters
    ----------
    image : array
        a multidimensional array representing the image
    
    Returns
    -------                      
    A dictionary mapping each dimension (x,t,y,z) to each corresponding value
    '''

     
    # Get the dimensions definition
    # Create a mapping from dimension types to axis indices
    dimensions_indices = {dim: i for i, dim in enumerate(dimensions_type)}
    
    # Map each dimension type to its size in the image shape
    dimensions_map = {dim: image.shape[dimensions_indices[dim]] for dim in dimensions_type}
    
    return dimensions_indices, dimensions_map
 
def make_folder (*folder_names):
    '''
    This function checks if the specified folders exist inside the global folder path.
    If a folder does not exist, it creates it.

    Parameters
    ----------
    folder_names : list of str 
        list of folder names to be created
        

    Returns
    -------
    None.

    '''
    for folder_name in folder_names:
        
        # Full path to the folder
        folder_path = os.path.join(folder, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            
            # Create the folder if it doesn't exist
            os.makedirs(folder_path)
            
def redefine_dimensions(dimension_to_remove):
    '''
    Create new dimensions and indices for the current slice.

    Parameters
    ----------
    dimension_to_remove : str
        The dimension to remove (e.g., 't').

    Returns
    -------
    tuple
        New dimensions map and indices.
    '''
    global dimensions_map
    global dimensions_indices
    
    temp_dimensions_map = {dim: size for dim, size in dimensions_map.items() if dim != dimension_to_remove}
    temp_dimensions_indices = {dim: i for i, dim in enumerate(temp_dimensions_map)}
    
    return temp_dimensions_map, temp_dimensions_indices


# End of locally-defined functions
# ===================================   
 

# Start of MAIN
# ============================
   
# Create a folder to store the results
make_folder('projections')


# Iterate over the files in folder
for file in sorted(os.listdir(folder)):
    
    if any (channel in file for channel in ch_name): # filter tiff files named as specified in ch_name list
        
        # Extract file names using defined function
        filepath, filename, savename,basename= filenaming(folder, file)
    
        # Import image
        img = import_image(filepath)
        
        # Get the image dimensions in the appropiate order
        dimensions_indices, dimensions_map= get_dimensions(img)
        
        # Check that the image contains z-stacks
        if "z" in dimensions_map:
        
            # For iamges containing a single time-point
            if "t" not in dimensions_map:
                
                sd, maxsd = focused_z(img, dimensions_indices) # Get an array with the sd per slice and the position of the maximal sd
                
                # Perform the projection and save
                img_proj= projection_3D(img, dimensions_indices)
                
                # Save the projected image
                img_path = os.path.join(folder,'projections', f"{basename}_{proj_type}.tif")
                imsave(img_path,img_proj)
            
            # For time-lapse images
            elif "t" in dimensions_map:
                
                # Start an empty array to concatenate the slices
                proj_timelapse = []
                
                # Start a for loop to iterate over each time-point
                for t in range(dimensions_map['t']):
                    
                    # Slice the image in the correct index
                    img_tslice = img.take(t,axis=dimensions_indices['t'])
                    
                    # Redefine dimensions for the specified dimenstion
                    s_dimensions_map, s_dimensions_indices =redefine_dimensions('t')
                    
                    # Obtain the sd info per each stack and the index of the best one
                    sd, maxsd = focused_z(img_tslice, s_dimensions_indices) # Get an array with the sd per slice and the position of the maximal sd
                    
                    # Perform the projection and return the projected image
                    img_proj_tp = projection_3D(img_tslice, s_dimensions_indices)
                    
                    # Concatenate slice to time-lapse array
                    # Append the projected slice to the list
                    proj_timelapse.append(img_proj_tp)
                
                # Concatenate the projected images along a new time axis
                img_stack = np.stack(proj_timelapse, axis=0)
                
                # Save the projected image
                img_path = os.path.join(folder,'projections', f"{basename}_{proj_type}.tif")
                imsave(img_path,img_stack)        
        
        else:
            raise ValueError(f"Unsupported image dimensions. {filename} does not contain z-stacks")
