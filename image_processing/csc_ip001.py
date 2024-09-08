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
    (brightfield or fluorescent).

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


"""


#library imports
import nd2
import os
import numpy as np
from tifffile import imsave, imread
import pandas as pd
import time
import matplotlib.pyplot as plt

# User-configurable parameters
# ============================
# Define the folder path containing the images
folder= '/Users/csalatca/Desktop/temporary_microscopyfiles/img_test_zproj/fluor'

# Define the type of projection to be applied: 'mean', 'max', or 'best'
proj_type='mean'

# Define the number of stacks to include above and below the best-focused plan
stack_step= 1 # it means it will get -n, best, +n 

# Define the channel type: BF for brightfield F for Fluoresencent images
ch_type= 'F'

# Define the channel name
ch_name= ['WL508'] # it can be a list to create projections of several fluorescent channels at a time

# Define the dimensions arrangement
dimensions_type= ('z','x','y') #t for time, z for z_stack, x and y dimensions


# End of user-configurable parameters
# ===================================

# Locally-defined functions
# ============================

def filenaming(folder,file):
    "This function gets the path and the name of each file in a directory"
    "and generates a new name for saving segmentation results in jpg "
    filepath = os.path.join(folder, file)
    filename = os.path.basename(filepath)
    savename = str(filename[:filename.index(".")])+'_'+str(proj_type)+'.tif'
    basename = str(filename[:filename.index(".")])
    #ovname=str(filename[:filename.index("_")])+'_coloc'+'.jpg'
    names=(filepath,filename,savename,basename)
    return names

def focused_z(image):
    "This function determines the most focused z slice for fluorescent channels within a multi-dimensional array"
    'to do so it calculates the frame with the highest standard deviation'
    sd=image.std(axis=(1,2)) #by axis we are telling it to flatten the first and second dimensions, so Y and X and get the sd of all the values
    maxsd=sd.argmax() 
    return sd, maxsd

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

def import_image(filepath):
    'This function checks wheter the image is a TIFF file or an ND2 and uses the appropiate code to import it'
    if filename.endswith('.tif'):
        img = imread(filepath)  # Import TIFF file
    elif filename.endswith('.nd2'):
        img = nd2.imread(filepath) # Import ND2 file
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return img

def projection_3D (img):
    'This function performs the desired projection on the image'
    global proj_type
    global maxsd
    global filename
    global dirpath
    global stack_step
    global dimensions
    global dimensions_map
    global dimensions_indices
    
    # Get variables from dimensions_map
    z = dimensions_map['z']
    
    
    # Get the two first blocks of filename 
    parts = filename.split('_')
    output_name = '_'.join(parts[:2])
    
    # Create a log file to store the information about the z-stack used
    log_file = os.path.join(dirpath, f"{output_name}_{proj_type}_log.txt")
    
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
        if (stack_depth) > len(dimensions):
            raise ValueError(f"Stack depth should be smaller or equal than image dimensions")
        
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
         
            # Log the z-stack used
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

        # Save the projected image
        img_path = os.path.join(dirpath, f"{output_name}_{proj_type}.tif")
        imsave(img_path,img_proj)        
    
    return None

'''        
        # to slice the image
        img_z = img.take(z_range, axis=dimension_indices['z']) # find the axis where z is stored
        
    
                
    
    # Save the projected image
    img_path = os.path.join(dirpath, f"{output_name}_{proj_type}.tif")
    imsave(img_path,img_proj)
        
    return None
 '''   

# End of locally-defined functions
# ===================================   
 

# Start of MAIN
# ============================
   
# Create a folder to store results
dirpath=os.path.join(folder,'projections')

# Check if the directory exists, if not, create it
isExist = os.path.exists(dirpath)
if not isExist:
        os.mkdir(dirpath)
        

# Iterate over the files in folder
for file in sorted(os.listdir(folder)):
    
    if any (channel in file for channel in ch_name): # filter tiff files named as specified in ch_name list
        
        # Extract file names using defined function
        filepath, filename, savename,basename= filenaming(folder, file)
    
        # Import image
        img = import_image(filepath)
        
        # Get image dimensions
        dimensions = img.shape
        
        # Get the dimensions definition
        # Create a mapping from dimension types to axis indices
        dimensions_indices = {dim: i for i, dim in enumerate(dimensions_type)}

        # Map each dimension type to its size in the image shape
        dimensions_map = {dim: img.shape[dimension_indices[dim]] for dim in dimensions_type}
    
        
        # For containing a single time-point
        if t not in dimension_map:
            
            sd, maxsd = focused_z(img) # Get an array with the sd per slice and the position of the maximal sd
            
            # Perform the projection and save
            projection_3D(img)
        
        
        
        
        elif len(dimensions) ==4:
        
        else:
            raise ValueError(f"Unsupported image dimensions: {dimensions}")

            
            
            
        
        
    










#I create a dictionary to store the data of the different images
report_images={}


count=0 

# Create a loop to process each file in the folder
for file in sorted(os.listdir(folder)):
    if file.endswith('.tif') and any(channel in file for channel in channels): #to filter for tiff files belonging to the channels specified in the channel list
          
          # I create lsits to store the data for the report
          timepoints_ls=[]
          nprojz_ls=[]
          projtype_ls=[]
          start_ls=[]
          stop_ls=[]
          
          # Extract file information using a custom function filenaming()
          filepath, filename, savename,bname= filenaming(folder, file)
          
          # Import the TIFF image
          img=imread(filepath)
          
          # Get the shape fo the image
          size= img.shape # for multidimensional files only containing multimple timepoints and z-stacks 'T', 'Z', 'Y' and 'X' are displayed                  
          
          if len(size)>3: #we check that image contains time-points
              count+=1
              img_proj_tp= np.empty((1,size[2],size[3]), dtype=np.uint16) #I create an empty array to allocate the results of the projection
              for i in range(size[0]):
                  img_stp=img.take(i, axis=0) #I generate a new image of only the current time-point
                  timepoints_ls.append(i+1) #I generate lists for the report, I add 1 so it can be compared to the file when visualized with Fiji
                  projtype_ls.append(proj)
                  if type(n)==range:
                      pass #add here the mean/ max projection with defined range
                  elif type (n)==int and file.__contains__('BF'):
                      #print('This is the BF image')
                      sd_df, minsddiff=focused_z_v2(img_stp) #to automatically detect the best focused stack for BF 
                      #print ('For time-point ' +str(i)+ ' the best focused z is: ' +str(minsddiff))
                      start=minsddiff-n #I calculate the starting stack for the projection
                      stop=minsddiff+n+1 #I calculate the starting stack for the projection, I add 1 so the last wanted z is included
                      nprojz=2*n+1 #I calculate the number of z-stacks that will be used for the projection
                      #print(start, stop, nprojz)
                      nprojz_ls.append(nprojz)
                  elif type (n)==int and 'BF' not in file:
                      #print('This is not the BF image')
                      sd, maxsd=focused_z(img_stp) #to automatically detect the best focused stack fro fluorescent channels
                      #print ('For time-point ' +str(i)+ ' the best focused z is: ' +str(minsddiff))
                      start=maxsd-n #I calculate the starting stack for the projection
                      stop=maxsd+n+1 #I calculate the starting stack for the projection, I add 1 so the last wanted z is included
                      nprojz=2*n+1 #I calculate the number of z-stacks that will be used for the projection
                      #print(start, stop, nprojz)
                      nprojz_ls.append(nprojz)
                  if start<0: #in case that the most focused z is not central and displaced towards the beginning, we get the first ones
                      indices=range(0,nprojz)
                  elif stop>size[1]:
                      indices=range(size[1]-nprojz,size[1])      
                  else:
                      indices=range(start,stop) 
                      #print(indices)
                      start_ls.append(list(indices)[0]+1) #I add 1 to make it comparable to the file visualized in Fiji
                      stop_ls.append(list(indices)[-1]+1) #I add 1 to make it comparable to the file visualized in Fiji
                      img_stp_z=img_stp.take(indices, axis=0) #I obtain the multi-dimensional image containing only the data to be projected
                  if proj=='mean':
                          #print ('mean projection is going to be performed')
                          img_proj=np.mean(img_stp_z,axis=0, dtype=np.uint16) #I perform mean projection
                  elif proj=='max':
                          #print ('max projection is going to be performed')
                          img_proj=np.max(img_stp_z,axis=0) #I perform the max projection
                  'I concatenate all the resulting projected time-points'
                  img_proj=np.expand_dims(img_proj, axis=0) #insert a new axis to allocate time at position 0
                  img_proj_tp=np.concatenate((img_proj_tp,img_proj),dtype=np.uint16)
              'we have generated and empty array that has to be removed, placed at the first position'
              'we take from 1 to the last position of axis 0'
              img_proj_tp=img_proj_tp.take(range(1,img_proj_tp.shape[0]), axis=0)
              savefile=os.path.join(dirpath,savename)
              imsave(savefile,img_proj_tp)
              #I create a report to know how the projection has been done
              proj_report={'Timepoint':timepoints_ls, 'Numb projected z':nprojz_ls,'Type of proj':projtype_ls,'Start z':start_ls, 'Stop z':stop_ls }
              report_images[bname]=proj_report
    
    # to open nd2 files and filter by the specified channels
    elif file.endswith('.nd2') and any(channel in file for channel in channels):
          print(file)
          
          # Get the info about the file using a cutom function filenaming ()
          filepath, filename, savename, bname= filenaming(folder, file)
          
          with nd2.ND2File(filepath) as ndfile:
              
              # Get the dimensions of the file [it returns a dict]
              size=ndfile.sizes #it returns a dictionary containing the dimensions of the file
             
              # Read the info in the size dict and assign them to single variables
              if size.__contains__('Z'):
                  z=size['Z']
                  X=size['X']
                  Y=size['Y']
                  T=size['T']
                  
                  # Filter for images with multiple z-stacks
                  if z>1: 
                      count+=1
                      
                      # Import the ND2 file ['T','Z', 'Y' and 'X' are displayed]
                      img= nd2.imread(filepath) 
                      
                      # Create an empty array with 1 for Time dimension and the same X, Y to store the projected data
                      img_proj_tp= np.empty((1,X,Y), dtype=np.uint16)
                      
                      # I create lsits to store the data for the report
                      timepoints_ls=[]
                      nprojz_ls=[]
                      projtype_ls=[]
                      start_ls=[]
                      stop_ls=[]
                  
                  # In case the image is a time-lapse    
                  if T>1:
                     
                      # To get individual images per each time-point
                      for t in range (T):
                          img_stp=img.take(t,axis=0) #flaw:change it so it corresponds always to the T axis independently of how the metadata is stored
                          timepoints_ls.append(t+1) #I generate lists for the report, I add 1 so it can be compared to the file when visualized with Fiji
                          projtype_ls.append(proj)
                          
                          '''
                          Missing code: in case we want to do the projection for a specific range '
                          if type(n)==range:
                              pass #add here the mean/ max projection with defined range
                          '''    
                          # To deterimine the best focused Z-frame for Brightfield images    
                          if isinstance (n,int) and 'BF' in filename:
                          
                              sd, minsddiff=focused_z_v2(img_stp) #to automatically detect the best focused stack
                              #print ('For time-point ' +str(i)+ ' the best focused z is: ' +str(minsddiff))
                              start=minsddiff-n #I calculate the starting stack for the projection
                              stop=minsddiff+n+1 #I calculate the starting stack for the projection, I add 1 so the last wanted z is included
                              nprojz=2*n+1 #I calculate the number of z-stacks that will be used for the projection
                              #print(start, stop, nprojz)
                              nprojz_ls.append(nprojz)
                              
                          # To determint the best focused z-frame for the fluorescent images not brightfield)
                          elif isinstance (n,int) and 'BF' not in filename:
                              #print('This is not the BF image')
                              sd, maxsd=focused_z(img_stp) #to automatically detect the best focused stack fro fluorescent channels
                              #print (f'For time-point {t} the best focused z is: {maxsd}')
                              
                              start=maxsd-n # To calculate the starting stack for the projection
                              stop=maxsd+n+1 # To calculate the starting stack for the projection, I add 1 so the last wanted z is included
                              nprojz=2*n+1 # To calculate the number of z-stacks that will be used for the projection
                              #print(start, stop, nprojz)
                              nprojz_ls.append(nprojz)
                          
                              
                          if start<0: #in case that the most focused z is not central and displaced towards the beginning, we get the first ones
                              indices=range(0,nprojz)
                          elif stop>z:
                              indices=range(z-nprojz,z)      
                          else:
                              indices=range(start,stop) 
                              #print(indices)
                          
                          start_ls.append(indices[0]+1) # I add 1 to make it comparable to the file visualized in Fiji
                          stop_ls.append(indices[-1]+1) 
                          img_stp_z=img_stp.take(indices, axis=0) #I obtain the multi-dimensional image containing only the data to be projected
                          
                          if proj=='mean':
                              #print ('mean projection is going to be performed')
                              img_proj=np.mean(img_stp_z,axis=0, dtype=np.uint16) #I perform mean projection
                          
                          elif proj=='max':
                              #print ('max projection is going to be performed')
                              img_proj=np.max(img_stp_z,axis=0) #I perform the max projection
                              
                          'I concatenate all the resulting projected time-points'
                          img_proj=np.expand_dims(img_proj, axis=0) #insert a new axis to allocate time at position 0
                          img_proj_tp=np.concatenate((img_proj_tp,img_proj),dtype=np.uint16)
                          
                      'we have generated and empty array that has to be removed, placed at the first position'
                      'we take from 1 to the last position of axis 0'
                      img_proj_tp=img_proj_tp.take(range(1,img_proj_tp.shape[0]), axis=0)
                      
                      # To save the image with the projection
                      savefile=os.path.join(dirpath,savename)
                      imsave(savefile,img_proj_tp)
                      
                      #I create a report to know how the projection has been done
                      proj_report={'Timepoint':timepoints_ls, 'Numb projected z':nprojz_ls,'Type of proj':projtype_ls,'Start z':start_ls, 'Stop z':stop_ls }
                      report_images[bname]=proj_report
                  elif len(size)==3 and file.__contains__('BF'): 
                     sd, minsddiff=focused_z_v2(img) #to automatically detect the best focused stack 
                     start=minsddiff-n #I calculate the starting stack for the projection
                     stop=minsddiff+n+1 #I calculate the starting stack for the projection, I add 1 so the last wanted z is included
                     nprojz=2*n+1 #I calculate the number of z-stacks that will be used for the projection
                     #print(start, stop, nprojz)
                     nprojz_ls.append(nprojz) 
                     projtype_ls.append(proj)
                     
                     if start<0: #in case that the most focused z is not central and displaced towards the beginning, we get the first ones
                         indices=range(0,nprojz)
                     elif stop>z:
                         indices=range(z-nprojz,z)      
                     else:
                         indices=range(start,stop) 
                     start_ls.append(list(indices)[0]+1) #I add 1 to make it comparable to the file visualized in Fiji
                     stop_ls.append(list(indices)[-1]+1) #I add 1 to make it comparable to the file visualized in Fiji
                     img_z=img.take(indices, axis=0) #I obtain the multi-dimensional image containing only the data to be projected
                     if proj=='mean':
                         #print ('mean projection is going to be performed')
                         img_proj=np.mean(img_z,axis=0, dtype=np.uint16) #I perform mean projection
                     elif proj=='max':
                         #print ('max projection is going to be performed')
                         img_proj=np.max(img_z,axis=0) #I perform the max projection
                     # to save the projection
                     savefile=os.path.join(dirpath,savename)
                     imsave(savefile,img_proj)
                  proj_report={'Numb projected z':nprojz_ls,'Type of proj':projtype_ls,'Start z':start_ls, 'Stop z':stop_ls }
                  report_images[bname]=proj_report
                  
#I define a path to store the report
reportpath=os.path.join(dirpath,'projection_report.xlsx') #I create name and path to store the reports corresponding to each file     
# Create an Excel writer object
writer = pd.ExcelWriter(reportpath)

# Iterate over the image_data dictionary and write each report to a separate sheet
for image_name, report_data in report_images.items():
    df = pd.DataFrame(report_data)
    df.to_excel(writer, sheet_name=image_name)

# Save the Excel file
writer.save()   

endtime=time.time()

print("Time of execution for",count,"files:",(endtime-starttime)/60, 'min')            
          
