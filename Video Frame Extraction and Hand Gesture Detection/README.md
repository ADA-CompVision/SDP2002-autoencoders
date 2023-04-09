Video Frame Extraction and Hand Gesture Detection

This Python script utilizes the OpenCV and MediaPipe libraries to process a video file, extract frames and detect hand gestures in the frames.

Features

Extract frames from a video file at a specified frames per second (FPS).
Save the extracted frames as images in a folder with a formatted filename based on the duration of the frame.
Utilize the MediaPipe Hand Detection model to detect hand gestures in the saved frames.
Print the detected hand gestures and their corresponding labels (e.g., "Right" or "Left") on the console.

Dependencies:
Python 3.x
OpenCV (4.5.4 or higher)
MediaPipe (0.8.13 or higher)

How to run:
pip install opencv-python mediapipe
Clone or download this repository.
Place the video file you want to process in the same directory as the script.
Open the script in a Python IDE or text editor and modify the parameters, such as video filename, desired FPS for frame extraction, and hand detection confidence threshold, according to your needs.
Run the script using a Python interpreter.
The script will create a folder with the name of the video file in the same directory, and save the extracted frames as images in that folder.
The detected hand gestures and their labels will be printed on the console for each saved frame.

python video_frame_extraction_and_hand_gesture_detection.py
