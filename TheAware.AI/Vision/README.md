
# Video Frame Processing, Text Extraction, and IP Webcam Setup

This Python script enables the processing of video frames, including the identification of stable frames, the enhancement of image quality, and efficient text extraction using the Google Cloud Vision API. Additionally, it provides guidelines for setting up IP Webcam on an Android device for live video streaming.

## Installation

Ensure you have the required dependencies installed. You can install them via pip:

```bash
pip install opencv-python google-cloud-vision
```

## IP Webcam Setup

To set up IP Webcam on your Android device, follow these steps:
Follow the link : [Connect your android camera to webcam](https://www.geeksforgeeks.org/connect-your-android-phone-camera-to-opencv-python/)


 
## Usage

Execute the script `main.py` with the following command-line arguments:

```bash
python main.py --credential <path_to_json_credentials> --output_directory <output_directory_path>
```

- `--credential`: Path to the JSON file containing Google Cloud credentials.
- `--output_directory`: Path to the output directory for extracted text.

## Script Overview

The script performs the following key tasks:

1. **Initialize KeyFramesProcessor**: Initializes the KeyFramesProcessor class with the provided Google Cloud credentials.

2. **Process Video Frames**: Scans the video frames to identify stable frames, which are likely to contain relevant information for text extraction.

3. **Enhance Image Quality**: Improves the quality of the stable frames to facilitate more accurate text extraction. Enhancement techniques may include contrast adjustment, noise reduction, and sharpness enhancement.

4. **Extract Text from Frames**: Utilizes the Google Cloud Vision API to extract text from the processed frames and saves the extracted text to a JSON file.

5. **Save Keyframes**: Saves the final keyframes to a designated folder for further analysis or reference.

## Constants

- `OUTPUT_JSON_FILE`: Name of the output JSON file containing extracted text.
- `KEYFRAMES_FOLDER`: Folder name for saving final keyframes.

## Requirements

- Python 3.6+
- OpenCV
- Google Cloud Vision API

## Example

```bash
python main.py --credential /path/to/credentials.json --output_directory /path/to/output_directory
```
## IP webcam setup 
Update the url variable of video_processing.py with the url obtained when you setup the IPWebcam on your android phone.

# video_processing.py file

```bash
url = "http://10.145.12.247:8080//shot.jpg"
```

## Notes

- Replace `/path/to/credentials.json` with the actual path to your Google Cloud credentials JSON file.
- The script saves the extracted text as a JSON file in the specified output directory.
- Ensure that you have access to the Google Cloud Vision API and that billing is enabled for your project.

## Authors

- Developed by Abhishek Tiwari

For additional information, refer to the [Google Cloud Vision API documentation](https://cloud.google.com/vision).

---

"# My_Internships" 
"# My_Internships" 
