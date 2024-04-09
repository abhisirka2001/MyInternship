# api.py
import os

def set_google_credentials(json_credentials_path):
    """
    Set the path to Google Cloud credentials.

    This function sets the environment variable 'GOOGLE_APPLICATION_CREDENTIALS' to the provided JSON file path.
    It is necessary to provide the path to the JSON file containing Google Cloud credentials before using Google Cloud services,
    such as the Vision API for text detection in this project.

    Parameters:
    - json_credentials_path (str): Path to the JSON file containing the Google Cloud credentials.
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_credentials_path


