from google.cloud import vision
import cv2
import json

# constants
JSON_INDENTATION = 2     # Number of spaces to use for JSON indentation
START_INDEX = 1          # so START_INDEX is set to 1 to align with common conventions.
TEXT_INDEX = 0           # Text string can be retrieved at the 0 index of text annotations variable
EXTENSION_IMAGE = '.jpg' # Extension for image files
ENCODING_NAME = 'utf-8'  # Encoding name for json files
WRITE_MODE ='w'          # Write mode for file operations

def process_images_with_vision(frames, output_json_file):
    """
    Extract text from a list of frames using Google Cloud Vision and write the results to a JSON file.

    Parameters:
    - frames (list): List of frames (images).
    - output_json_file (str): Output JSON file path.

    Returns:
    - None
    """
    # Create a list to store the results for each frame
    results = []

    # Iterate through the frames
    for i, current_frame in enumerate(frames, start=START_INDEX):

        # Create a Google Cloud Vision client
        vision_client = vision.ImageAnnotatorClient()
        image = vision.Image()

        # Convert the frame to bytes
        _, image_content = cv2.imencode(EXTENSION_IMAGE, current_frame)
        image.content = image_content.tobytes()

        # Perform text detection
        response = vision_client.text_detection(image=image)
        text_annotations = response.text_annotations

        # If text is detected, add the result to the list
        if text_annotations:
            text = text_annotations[TEXT_INDEX].description
            result = {"page": i, "text": text}
        else:
            result = {"page": i, "text": "No text detected"}

        results.append(result)

    # Write the results to the JSON file
    with open(output_json_file, WRITE_MODE, encoding=ENCODING_NAME) as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=JSON_INDENTATION)
