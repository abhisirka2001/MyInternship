import cv2
import numpy as np
from skimage import filters
from scipy.ndimage import interpolation as inter
import matplotlib.pyplot as plt

CLIP_LIMIT = 2
ZERO_INDEX = 0
MAX_PIXEL = 255
MIN_PIXEL = 0
ALL_CONTOURS_INDEX = -1
CONTOUR_APPROXIMATION = 0.02
SCORE_FACTOR = 2
ONE_INDEX= 1
X_SCALE = 0.75
Y_SCALE = 0.75
LIMIT = 45
CHANNEL = 2
GRID_SIZE = (4,4)

def binarize_images(frames):
    
    """
    Binarize the input images using Otsu's thresholding method and crop the largest contour.

    Parameters:
    - frames (list of numpy.ndarray): List of input frames.

    Returns:
    - list of numpy.ndarray: List of cropped binary frames.
    """
    cropped_frames = []

    for frame in frames:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(img_gray, MIN_PIXEL, MAX_PIXEL, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables for the contour with maximum area
        max_contour_area = ZERO_INDEX
        max_contour_index = ALL_CONTOURS_INDEX

        # Iterate through all contours to find the one with the maximum area
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if contour_area > max_contour_area:
                max_contour_area = contour_area
                max_contour_index = i

        # Approximate the contour to a rectangle
        epsilon = CONTOUR_APPROXIMATION * cv2.arcLength(contours[max_contour_index], True)
        approx = cv2.approxPolyDP(contours[max_contour_index], epsilon, True)

        # Find the bounding rectangle of the approximated contour
        x, y, w, h = cv2.boundingRect(approx)

        # Crop out the region from the original image
        cropped_region = frame[y:y+h, x:x+w]
        cropped_frames.append(cropped_region)

    return cropped_frames

def findScore(img, angle):
    """
    Generates a score for the binary image received dependent on the determined angle.
    
    Parameters:
    - img (numpy.ndarray): Binary image array.
    - angle (float): Predicted angle at which the image is rotated by.
    
    Returns:
    - histogram of the image
    - score of potential angle
    """
    data = inter.rotate(img, angle, reshape=False, order=ZERO_INDEX)
    hist = np.sum(data, axis=ONE_INDEX)
    score = np.sum((hist[ONE_INDEX:] - hist[:ALL_CONTOURS_INDEX]) ** SCORE_FACTOR)
    return hist, score

def skewCorrect(frames):
    """
    Takes in a list of frames and determines the skew angle of the text, then corrects the skew and returns the corrected frames.
    
    Parameters:
    - frames (list of numpy.ndarray): List of input frames.
    
    Returns:
    - list of numpy.ndarray: List of corrected frames.
    """
    corrected_frames = []
    for frame in frames:
        # Resize the image
        img = cv2.resize(frame, (ZERO_INDEX, ZERO_INDEX), fx=X_SCALE, fy=Y_SCALE)

        delta = ONE_INDEX
        limit = LIMIT
        angles = np.arange(-limit, limit + delta, delta)
        scores = []
        for angle in angles:
            hist, score = findScore(img, angle)
            scores.append(score)
        bestScore = max(scores)
        bestAngle = angles[scores.index(bestScore)]
        rotated = inter.rotate(img, bestAngle, reshape=False, order=ZERO_INDEX)
        # Convert the image to grayscale if it's in color
        if len(rotated.shape) > CHANNEL:
            rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        # Ensure the image is in 8-bit unsigned integer format
        if rotated.dtype != np.uint8:
            rotated = rotated.astype(np.uint8)
        corrected_frames.append(rotated)
    
    return corrected_frames

def apply_clahe(frames):

    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) transformation to the list of frames.

    Parameters:
    - frames (list of numpy.ndarray): List of input frames.

    Returns:
    - list of numpy.ndarray: List of frames after applying CLAHE transformation.
    """

    clahe_frames = []
    # Apply CLAHE transform to each frame
    for frame in frames:
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=GRID_SIZE)
        # Apply CLAHE transform
        clahe_frame = clahe.apply(frame)
        clahe_frames.append(clahe_frame)

    return clahe_frames
