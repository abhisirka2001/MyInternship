import cv2
import numpy as np
import os
import time 
import requests
import numpy as np
import imutils

# Constants
# Your IP webcam URL. Make sure to add "/shot.jpg" at the end.
url = "http://10.145.12.247:8080//shot.jpg"

# Constants
THRESHOLD = 128                           # Threshold for binarization
KERNEL_LEN_DIVISOR = 100                  # Divisor for calculating kernel length
MIN_CONTOUR_AREA = 20000                  # Minimum contour area to be considered as text region
INITIAL_BOUNDING_BOX = (20, 20, 750, 750) # Initial bounding box coordinates
COORDINATE_SMOOTHING_FACTOR = 0.1         # Smoothing factor for text region coordinates
ADJUSTMENT_OFFSET = 5                     # Offset for adjusting coordinates
ADJUSTMENT_SIZE = 10                      # Size for adjusting coordinates
MORPH_KERNEL_HEIGHT = 20                  # Height of the morphological kernel for removing horizontal lines
ERODE_ITERATIONS = 3                      # Number of iterations for erosion in morphological operations
DILATE_ITERATIONS = 5                     # Number of iterations for dilation in morphological operations
CANNY_THRESHOLD1 = 90                     # Lower threshold for Canny edge detection
CANNY_THRESHOLD2 = 150                    # Upper threshold for Canny edge detection
MAX_PIXEL = 255                           # Maximum pixel intensity
TEXT_OFFSET_LENGTH = 10                   # Offset length for placing text above the bounding box
MIN_PIXEL = 0                             # Minimum pixel intensity
TEXT_THICKNESS = 2                        # Thickness of text in drawing
TEXT_REGION_SCALE = 0.5                   # Scaling factor for the font size in drawing text region information
KERNEL_SIZE = 1                           # Determines the shape of the kernel as (1,1)
MAX_AREA_INDEX = 0                        # Index of the contours which has the highest area
ZERO_INDEX = 0                            # Represents 0 while indexing
ONE_INDEX = 1                             # Represents 1 while indexing
TERMINATE_KEY = 'q'                       # Key which terminates the video processing
TEXT_REGION = "STABLE TEXT FRAME DETECTED SUCCESSFULLY" # String to be shown at the top of the bounding box drawn over the text
FRAME_NAME = "Processed Frame"            # Constant to store the name of the processing frame being displayed
OUTPUT_FOLDER_NAME ="scanned_frames"      # Output folder for storing the cropped frames after video processing
STABILITY_THRESHOLD = 5
PYR_SCALE = 0.5                           # Scale factor for building the image pyramid
LEVELS = 3                                # Number of pyramid layers including the initial image
WINSIZE = 15                              # Size of the averaging window used for optical flow
ITERATIONS = 3                            # Number of iterations the algorithm does at each pyramid level
POLY_N = 5                                # Size of the pixel neighborhood used to find polynomial expansion in each pixel
POLY_SIGMA = 1.2                          # Standard deviation of the Gaussian that is used to smooth derivatives
FLAGS = 0                                 # Operation flags; should be 0 for the Farneback algorithm
STABLE_FRAME_DURATION = 1.5               # Duration of stable frame required for saving frames (in seconds)
MULTIPLICATION_FACTOR = 3                 # Multiplication factor 
TEXT_TO_DISPLAY = "LOOKING FOR STABLE TEXT REGION"  # Text to display when no stable bounding box is detected
TEXT_COLOR = (0, 0, 255)                  # Red color for the text
FONT_SCALE = 1                            # Scale of the font
FONT_THICKNESS = 2                        # Thickness of the font
START_FRAME_START_TIME = 0                # Starting time while scanning a particular page
START_FRAME_DURATION = 0                  # Start duration while scanning a page
RESIZED_WIDTH = 1000                      # Width to which frames are resized
RESIZED_HEIGHT = 1800                     # Height to which frames are resized
SAVED_FRAME_MESSAGE = "Frame Saved"       # Message to indicate successful frame saving
SQUARE_FACTOR = 2                         # Factor to determine the size of a square
CENTERING_FACTOR = 2                      # Factor used for centering calculations
ALL_DIMENSIONS = -1                       # Constant indicating all dimensions

# Variables for tracking stable frame duration
stable_frame_start_time = 0  # Start time of the stable frame
stable_frame_duration = 0     # Duration of the stable frame


# Check if the folder exists else create new one
os.makedirs(OUTPUT_FOLDER_NAME,exist_ok=True)

def remove_horizontal_lines(image):
    """
    Remove horizontal lines from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - numpy.ndarray: Image without horizontal lines.
    """
    # Thresholding
    _ , image_bin = cv2.threshold(image, THRESHOLD, MAX_PIXEL, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_inv = MAX_PIXEL - image_bin

    # Morphological operations
    kernel_len = np.array(image).shape[ONE_INDEX] // KERNEL_LEN_DIVISOR
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, MORPH_KERNEL_HEIGHT))
    image_inv = cv2.erode(image_inv, horizontal_kernel, iterations=ERODE_ITERATIONS)
    horizontal_lines = cv2.dilate(image_inv, horizontal_kernel, iterations=DILATE_ITERATIONS)

    # Subtract horizontal lines from the image
    image_without_horizontal_lines = cv2.subtract(MAX_PIXEL * np.ones_like(image), horizontal_lines)

    return image_without_horizontal_lines


def process_video_frames():
    """
    Process video frames to detect and crop text regions.

    Parameters:
    - video_path (str): Path to the video file.

    Returns:
    - list: List of cropped frames containing text regions.
    """
        
    global stable_frame_start_time, stable_frame_duration  # Update to use global variables

    prev_gray = None
    frame_count = ZERO_INDEX
    prev_x, prev_y, prev_w, prev_h = INITIAL_BOUNDING_BOX
    cropped_frames = []

    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, ALL_DIMENSIONS)
        frame = imutils.resize(img, width=RESIZED_WIDTH, height=RESIZED_HEIGHT)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # In the process_video_frames function
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, PYR_SCALE, LEVELS, WINSIZE, ITERATIONS, POLY_N, POLY_SIGMA, FLAGS)
            magnitude = np.sqrt(flow[...,ZERO_INDEX]**SQUARE_FACTOR + flow[...,ONE_INDEX]**SQUARE_FACTOR)

            if np.mean(magnitude) < STABILITY_THRESHOLD:
                edges = cv2.Canny(gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
                new_image = remove_horizontal_lines(edges)

                contours, _ = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                if contours:
                    max_contour = contours[MAX_AREA_INDEX]
                    if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:
                        x, y, w, h = cv2.boundingRect(max_contour)

                        x = int((ONE_INDEX - COORDINATE_SMOOTHING_FACTOR) * prev_x + COORDINATE_SMOOTHING_FACTOR * x)
                        y = int((ONE_INDEX - COORDINATE_SMOOTHING_FACTOR) * prev_y + COORDINATE_SMOOTHING_FACTOR * y)
                        w = int((ONE_INDEX - COORDINATE_SMOOTHING_FACTOR) * prev_w + COORDINATE_SMOOTHING_FACTOR * w)
                        h = int((ONE_INDEX - COORDINATE_SMOOTHING_FACTOR) * prev_h + COORDINATE_SMOOTHING_FACTOR * h)

                        x -= ADJUSTMENT_OFFSET
                        y -= ADJUSTMENT_OFFSET
                        w += ADJUSTMENT_SIZE
                        h += ADJUSTMENT_SIZE

                        x = max(ZERO_INDEX, x)
                        y = max(ZERO_INDEX, y)
                        w = min(frame.shape[ONE_INDEX] - ONE_INDEX, w)
                        h = min(frame.shape[ZERO_INDEX] - ONE_INDEX, h)
                        # Coordinates are stable, append the cropped frame
                        cropped_frame = frame[y:y + h, x:x + w]
                        # Update stable frame duration
                        current_time = time.time()
                        if stable_frame_start_time == START_FRAME_START_TIME:
                            stable_frame_start_time = current_time
                        stable_frame_duration = current_time - stable_frame_start_time

                        # Check if stable frame duration is sufficient
                        if stable_frame_duration >= STABLE_FRAME_DURATION:
                            # Save the cropped frame with the current bounding box
                            frame_count += ONE_INDEX
                            filename = os.path.join(OUTPUT_FOLDER_NAME, f"page_number_{frame_count}.jpg")
                            cv2.imwrite(filename, cropped_frame)
                            cropped_frames.append(cropped_frame)
                            print(f"Page {frame_count} saved.")
                            # Display a message on the frame indicating that the frame has been saved
                            cv2.putText(frame, SAVED_FRAME_MESSAGE, (TEXT_OFFSET_LENGTH, TEXT_OFFSET_LENGTH * MULTIPLICATION_FACTOR), cv2.FONT_HERSHEY_SIMPLEX, TEXT_REGION_SCALE, (ZERO_INDEX, MAX_PIXEL, ZERO_INDEX), TEXT_THICKNESS)
                            stable_frame_start_time = START_FRAME_START_TIME  # Reset stable frame start time
                            stable_frame_duration = START_FRAME_DURATION    # Reset stable frame duration

                        prev_x, prev_y, prev_w, prev_h = x, y, w, h

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (MIN_PIXEL, MAX_PIXEL, MIN_PIXEL), TEXT_THICKNESS)
                        cv2.putText(frame, TEXT_REGION, (x, y - TEXT_OFFSET_LENGTH), cv2.FONT_HERSHEY_SIMPLEX, TEXT_REGION_SCALE, (MIN_PIXEL, MAX_PIXEL, MIN_PIXEL), TEXT_THICKNESS)

                         
            else:       
                # Reset stable frame duration if not stable
                stable_frame_start_time = START_FRAME_START_TIME
                stable_frame_duration = START_FRAME_DURATION

                # Calculate text size for centering
                text_size = cv2.getTextSize(TEXT_TO_DISPLAY, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[ZERO_INDEX]
                text_x = (frame.shape[ONE_INDEX] - text_size[ZERO_INDEX]) // CENTERING_FACTOR  # X coordinate for centering text
                text_y = (frame.shape[ZERO_INDEX] + text_size[ONE_INDEX]) // CENTERING_FACTOR  # Y coordinate for centering text

                # Display text at the center of the frame
                cv2.putText(frame, TEXT_TO_DISPLAY, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)


        cv2.imshow(FRAME_NAME, frame)
        
        # Check if 'p' key is pressed
        key = cv2.waitKey(ONE_INDEX) & 0xFF

        if key == ord(TERMINATE_KEY):
            break

        prev_gray = gray.copy()

    cv2.destroyAllWindows()

    return cropped_frames