import argparse
import os
import cv2

from video_to_frames.video_processing import process_video_frames
from text_extraction.api.vision_api import set_google_credentials
from text_extraction.image_to_text import process_images_with_vision
from frame_enhancement.frame_enhancement import apply_clahe
from frame_enhancement.frame_enhancement import binarize_images
from frame_enhancement.frame_enhancement import skewCorrect


# Constants
OUTPUT_JSON_FILE = "extracted_text.json"

KEYFRAMES_FOLDER ="final_keyframes"
ONE_INDEX = 1

class KeyFramesProcessor:
    """
    A class for processing video frames, extracting keyframes, and detecting text using Google Cloud Vision API.
    """

    def __init__(self, json_credentials_path):
        """
        Initialize the KeyFramesProcessor.

        Parameters:
        - json_credentials_path (str): Path to the JSON file containing Google Cloud credentials.
        """
        set_google_credentials(json_credentials_path)

    def process_video(self):
        """
        Process video frames, select keyframes, and return the final keyframes.

        Returns:
        - list: List of final keyframes.
        """
        scanned_frames = process_video_frames()
        return scanned_frames
    

    def binarize(self, frames):
        """
        Binarizes a list of input frames.

        Parameters:
        - frames (list of numpy.ndarray): List of input frames.

        Returns:
        - list of numpy.ndarray: List of binarized images.
        """
        cropped_images = binarize_images(frames)
        return cropped_images
    
    def clahe_transform(self, frames):
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) transform to a list of input frames.

        Parameters:
        - frames (list of numpy.ndarray): List of input frames.

        Returns:
        - list of numpy.ndarray: List of frames after applying CLAHE transform.
        """
        clahe_images = apply_clahe(frames)
        return clahe_images

    def skewcorrection(self, frames):
        """
        Performs skew correction on a list of input frames.

        Parameters:
        - frames (list of numpy.ndarray): List of input frames.

        Returns:
        - list of numpy.ndarray: List of frames after skew correction.
        """
        corrected_images = skewCorrect(frames)
        return corrected_images
    

    def process_frames_to_text(self, frames, output_json_file):
        """
        Extract text from a list of frames using Google Cloud Vision and save it to a json file.

        Parameters:
        - frames (list): List of frames (keyframes).
        - output_text_file (str): Output text file path.

        Returns:
        - None
        """
        process_images_with_vision(frames, output_json_file)
        

    def run_processing_pipeline(self, output_directory):
        """
        Run the complete processing pipeline to extract text from video frames.

        Parameters:
        - output_directory (str): Path to the output directory for extracted text.

        Returns:
        - None
        """
        # Step 1: Process video frames and extract keyframes
        scanned_frames = self.process_video()

        # Check if the output directory exists, create it if not
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        cropped_text_frames = self.binarize(scanned_frames)

        corrected_frames = self.skewcorrection(cropped_text_frames)

        final_enhanced_frames = self.clahe_transform(corrected_frames)

        # Step 2: Extract text from final keyframes using Google Cloud Vision
        output_json_file = os.path.join(output_directory, OUTPUT_JSON_FILE)


        self.process_frames_to_text(final_enhanced_frames, output_json_file)

        self.save_keyframes(final_enhanced_frames)

        print(f"Text extraction completed. Extracted text saved to: {output_json_file}")
    
    def save_keyframes(self, keyframes):
        """
        Save the final keyframes to a designated folder.

        Parameters:
        - keyframes (list): List of final keyframes.

        Returns:
        - None
        """
        keyframes_folder = os.path.join(os.getcwd(), KEYFRAMES_FOLDER)
        os.makedirs(keyframes_folder, exist_ok=True)

        for index, frame in enumerate(keyframes, start=ONE_INDEX):
            frame_filename = f"keyframe_{index}.jpg"
            frame_path = os.path.join(keyframes_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

        print(f"Final keyframes saved to: {keyframes_folder}")


def main():
    parser = argparse.ArgumentParser(description="Key Frames Extraction and Text Detection")
    parser.add_argument("--credential", required=True, help="Path to the JSON file containing Google Cloud credentials")
    parser.add_argument("--output_directory", required=True, help="Path to the output directory for extracted text")

    args = parser.parse_args()
    # Create the KeyFramesProcessor instance
    keyframes_processor = KeyFramesProcessor(args.credential)

    # Run the complete processing pipeline
    keyframes_processor.run_processing_pipeline(args.output_directory)

if __name__ == "__main__":
    main()

