#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
input_path = "C:\\Users\\User\\Downloads\\Dewinter2023\\Final_yolov5x\\image_sample\\2022-12-26-17-03-49.jpg"
bgr=  cv2.imread(input_path)
bgr_gray=  cv2.imread(input_path,0)
bgr_copy_2 = bgr.copy()
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
img_hsv = cv2.cvtColor(lab, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the green color in the HSV color space
lower_green = np.array([40, 130, 170])
upper_green = np.array([90, 255, 255])

# Threshold the HSV image based on the color range
img_mask = cv2.inRange(img_hsv, lower_green, upper_green)

#split into channels
b,g,r = cv2.split(bgr)
#apply adaptive thresholding
binary_image = cv2.adaptiveThreshold(bgr_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 2)
# Fill holes
filled_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
img_mask= cv2.dilate(img_mask,kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),iterations=2)
# subtracting the wbc and platelets mask from the original image
filled_image =(filled_image)-img_mask
# External contours of rbc
contours_ext,hierarchy = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_data_list_1 =[]

# empty dictionary for rbc,wbc, platelets count
Count_1={"RBC":0,"Platelet":0,"WBC":0}

contours_new = []

# Get the dimensions of the image
image_height, image_width = bgr.shape[:2]

# Create a list to store the non-boundary contours
non_boundary_contours = []

# Iterate through all the contours
for contour in contours_ext:
    min_contour_area = 250  # Minimum threshold area for rbc
    max_contour_area = 2000
    # Flag to check if contour touches the boundaries
    is_boundary = False
    # Approximate the contour to a closed polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
   
    # Calculate the contour area
    area = cv2.contourArea(approx)
    x, y, w, h = cv2.boundingRect(approx)
    # Iterate through all the points in the contour
    for point in contour:

        # Check if any point is touching the boundaries of the image
        x, y = point[0]       
        if x == 0 or x == image_width - 1 or y == 0 or y == image_height - 1:
              is_boundary=True 
              break 
    # Break the inner loop if any point touches the boundary
    if area >= min_contour_area and area < max_contour_area:
   # If the contour is not touching the boundary, add it to the non-boundary contours list
        if not is_boundary:
           cv2.drawContours(bgr_copy_2,[contour],-1,(0,255,0),2)
 

for contour in contours_ext:
    # Approximate the contour to a closed polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    min_contour_area = 250  # Minimum threshold area for rbc
    max_contour_area = 2000
    # Calculate the contour area
    area = cv2.contourArea(approx)
    x, y, w, h = cv2.boundingRect(approx)

    if area >= min_contour_area and area < max_contour_area:
        M = cv2.moments(approx)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        Count_1["RBC"] += 1
        # Add the contour number as text
        contour_number = str(Count_1["RBC"])
        contours_new.append(approx)
        # Add the contour data to the DataFrame
        # Inside the loop, create a dictionary of the contour data
        contour_dict = {"Contour Number": contour_number,
                "Contour": approx,
                "Area": area,
                "Perimeter": cv2.arcLength(approx, True)}
        contour_data_list_1.append(contour_dict)
        # Create a DataFrame from the list of dictionaries
        contour_data_1 = pd.DataFrame(contour_data_list_1)
        #cv2.drawContours(bgr_copy_1, [contour], -1, (255, 0, 0), 2)
rbc_count =Count_1["RBC"]
Count_1["RBC"]=0
outer_rbc_area = np.sum([cv2.contourArea(contour) for contour in contours_new])
# Internal and external contours of rbc
contours, hierarchy = cv2.findContours(filled_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

inner_rbc_area = 0
min_contour_area = 30
for i in range(len(contours)):
    contour = contours[i]
    contour_hierarchy = hierarchy[0][i]
    # Calculate the perimeter of the contour
    
    # Calculate the contour area
    area = cv2.contourArea(contour)
    if area >15:
            
            if contour_hierarchy[3] != -1:
                Count_1["RBC"]+=1
                # Inner contour (Has parent contour)
                inner_rbc_area += area
                
                #cv2.drawContours(bgr_copy_2,[contour],-1,(255,0,0),2)
total_rbc = (Count_1["RBC"]+rbc_count)/2
print("Red sample Blood Report :")
print("Total rbc:",total_rbc)

print("haemoglobin perecentage :Red pigment in the image ",(outer_rbc_area-inner_rbc_area)*100 /outer_rbc_area )

# calculating the MCV of RBCs
"""MCV stands for Mean Corpuscular Volume, and it is a measure of the average volume or size of red blood cells (erythrocytes) in a blood sample.
 It is typically reported in units of femtoliters (fL)."""
def calculate_mcv(contours):
    total_diameter = 0
    num_rbc = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 250:  # Filter out small contours
            num_rbc += 1
            (x, y), (w, h), _ = cv2.fitEllipse(contour)
            diameter = max(w, h)
            total_diameter += diameter
    
    mean_diameter = total_diameter / num_rbc
    mcv = (np.pi / 6) * (mean_diameter**3)
    return mcv

MCV = calculate_mcv(contours_new)
print("MCV :",MCV)

# Function to calculate coarseness
def calculate_coarseness(contour):
    perimeter = cv2.arcLength(contour, True)
    coarseness = perimeter / (2 * np.pi)
    return coarseness

# Function to calculate circularity
def calculate_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity

# calculating RDW of rbcs
"""variation in size of red blood cells (erythrocytes) in a blood sample. The RDW value is typically reported as a percentage.
lower rdw less variation in size"""
def calculate_rdw(contours):
    rbc_volumes = []

    for contour in contours:
        contour= cv2.convexHull(contour)
        area = cv2.contourArea(contour)
        radius = np.sqrt(area / np.pi)
        volume = (4/3) * np.pi * radius**3
        rbc_volumes.append(volume)

    mean_cell_volume = np.mean(rbc_volumes)
    rdw = np.std(rbc_volumes) / mean_cell_volume *10
    return rdw
rdw = calculate_rdw(contours_new)
print("RDW:", rdw)

# Finding count  of  WBC and platelets
contours_platelets,_ = cv2.findContours(img_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area =800
for contour in contours_platelets:
    area = cv2.contourArea(contour)
    if area <min_contour_area:
       #cv2.drawContours(bgr_copy_1,[contour],-1,(0,255,0),2)
       Count_1["Platelet"]=Count_1["Platelet"]+1
    else:
        #cv2.drawContours(bgr_copy_1,[contour],-1,(0,0,255),2)
        Count_1["WBC"]=Count_1["WBC"]+1
plt.figure(figsize=(10,10))
#plt.imshow(bgr_copy_1)
print("Total WBC :",Count_1["WBC"])
print("Total Platelets :",Count_1["Platelet"])


contour_data_1["Coarseness"] = contour_data_1["Contour"].apply(calculate_coarseness)
# Calculate additional features for each contour
contour_data_1["Circularity"] = contour_data_1["Contour"].apply(calculate_circularity)
contour_data=contour_data_1.copy()
#Choosing Normal RBC which are circular in shape and have normal size variation
filtered_normal_contours = contour_data[(contour_data["Coarseness"]>14) & (contour_data["Coarseness"]<21)&(contour_data["Circularity"]>0.80)]
print("Normal RBC :",len(filtered_normal_contours))
print("percentage normal :",len(filtered_normal_contours)*100/total_rbc)
normal = filtered_normal_contours["Contour"].values
cv2.imshow("blue sample rbc",bgr_copy_2)
cv2.waitKey(0)

'''------------------------------------------------BLUE SAMPLE--------------------------------------------------------------------------'''


input_path = "C:\\Users\\User\\Downloads\\Dewinter2023\\Final_yolov5x\\image_sample\\2022-10-19-15-03-24.jpg"
bgr=  cv2.imread(input_path)
bgr_gray=cv2.imread(input_path,0)
crop = bgr
bgr_copy_1=bgr.copy()
b,g,r = cv2.split(crop)
# Perform histogram equalization on the green channel
g_equalized = cv2.equalizeHist(g)

# Perform contrast stretching
min_intensity = 100
max_intensity = 255
stretched = cv2.normalize(g_equalized, None, min_intensity, max_intensity, cv2.NORM_MINMAX)

# Apply adaptive thresholding
binary_image = cv2.adaptiveThreshold(stretched, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 2)


# Fill holes
lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
img_hsv = cv2.cvtColor(lab, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the green color in the HSV color space
lower_green = np.array([40, 130, 170])
upper_green = np.array([90, 255, 255])

# Threshold the HSV image based on the color range
img_mask = cv2.inRange(img_hsv, lower_green, upper_green)
img_mask_new = cv2.dilate(img_mask,kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),iterations=2)
#filled_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
filled_image = cv2.erode(binary_image,kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),iterations=1)
filled_image_new = filled_image-img_mask_new


# External contours of rbc
contours_ext,hierarchy = cv2.findContours(filled_image_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_data_list_2 = []

# empty dictionary for rbc,wbc, platelets count
Count_1={"RBC":0,"Platelet":0,"WBC":0}
min_contour_area = 50
max_contour_area = 5000
contours_new = []
# Get the dimensions of the image
image_height, image_width = bgr.shape[:2]

# Create a list to store the non-boundary contours
non_boundary_contours = []

# Iterate through all the contours
for contour in contours_ext:
    # Flag to check if contour touches the boundaries
    is_boundary = False
        # Approximate the contour to a closed polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
   
    # Calculate the contour area
    area = cv2.contourArea(approx)
    x, y, w, h = cv2.boundingRect(approx)

    # Iterate through all the points in the contour
    for point in contour:
        # Check if any point is touching the boundaries of the image
        x, y = point[0]
        
        if x == 0 or x == image_width - 1 or y == 0 or y == image_height - 1:
              is_boundary = True
              break  # Break the inner loop if any point touches the boundary
        '''if area >min_contour_area and area < max_contour_area:
            cv2.drawContours(bgr_copy_1,[contour],-1,(0,255,0),2)'''
    # If the contour is not touching the boundary, add it to the non-boundary contours list
    if area > min_contour_area and area < max_contour_area:
       if not is_boundary:
          cv2.drawContours(bgr_copy_1,[contour],-1,(0,255,0),2)
          non_boundary_contours.append(contour)

for contour in contours_ext:
    # Approximate the contour to a closed polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
   
    # Calculate the contour area
    area = cv2.contourArea(approx)
    x, y, w, h = cv2.boundingRect(approx)

    if area >= min_contour_area and area < max_contour_area:

        Count_1["RBC"] += 1
        # Add the contour number as text
        contour_number = str(Count_1["RBC"])
        contours_new.append(approx)
        text_size, _ = cv2.getTextSize(contour_number, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Add the contour data to the DataFrame
        contour_data_list_2.append({"Contour Number": contour_number,"Contour" : approx,"Area": area, "Perimeter": cv2.arcLength(approx, True)})
        contour_data_2 = pd.DataFrame(contour_data_list_2)
        #cv2.drawContours(bgr_copy_1, [contour], -1, (255, 0, 0), 2)
rbc_count =Count_1["RBC"]
Count_1["RBC"]=0
outer_rbc_area = np.sum([cv2.contourArea(contour) for contour in contours_new])
# Internal and external contours of rbc
contours, hierarchy = cv2.findContours(filled_image_new, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

inner_rbc_area = 0
min_contour_area = 30
for i in range(len(contours)):
    contour = contours[i]
    contour_hierarchy = hierarchy[0][i]
    # Calculate the perimeter of the contour
    
    # Calculate the contour area
    area = cv2.contourArea(contour)
    if area >15:
            
            if contour_hierarchy[3] != -1:
                Count_1["RBC"]+=1
                # Inner contour (Has parent contour)
                inner_rbc_area += area
                
                #cv2.drawContours(bgr_copy_1,[contour],-1,(255,0,0),2)
total_rbc = (Count_1["RBC"]+rbc_count)/2
print("\nBLUE SAMPLE BLOOD REPORT :")
print("Total rbc:",total_rbc)
print("haemoglobin perecentage :",((outer_rbc_area-inner_rbc_area))*100 /outer_rbc_area )


MCV = calculate_mcv(contours_new)
print("MCV :",MCV)

rdw = calculate_rdw(contours_new)
print("RDW:", rdw)


contour_data_2["Coarseness"] = contour_data_2["Contour"].apply(calculate_coarseness)
# Calculate additional features for each contour
contour_data_2["Circularity"] = contour_data_2["Contour"].apply(calculate_circularity)
contour_data=contour_data_2.copy()
#Choosing Normal RBC which are circular in shape and have normal size variation
filtered_normal_contours = contour_data[(contour_data["Coarseness"]>14) & (contour_data["Coarseness"]<21)&(contour_data["Circularity"]>0.80)]
print("Normal RBC :",len(filtered_normal_contours))
print("percentage normal :",len(filtered_normal_contours)*100/total_rbc)
normal = filtered_normal_contours["Contour"].values
# Display the mask image with only circular contours
min_contour_area = 30
max_contour_area = 5000

#cv2.drawContours(crop_copy, contours_red, -1, (255, 0, 0), 2)
contours_platelets,_ = cv2.findContours(img_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area =1000
for contour in contours_platelets:
    area = cv2.contourArea(contour)
    if area >min_contour_area:
        #cv2.drawContours(crop_copy,[contour],-1,(0,0,255),2)
        Count_1["WBC"]=Count_1["WBC"]+1
    else:
        #cv2.drawContours(crop_copy,[contour],-1,(0,255,0),2)
        Count_1["Platelet"]=Count_1["Platelet"]+1
  
print("Total WBC :",Count_1["WBC"])
print("Total Platelets :",Count_1["Platelet"])
cv2.imshow("blue sample rbc",bgr_copy_1)
cv2.waitKey(0)


