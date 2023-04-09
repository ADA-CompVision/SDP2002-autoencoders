Video Frame Extraction and Hand Gesture Detection

This Python script utilizes the OpenCV and MediaPipe libraries to process a video file, extract frames and detect hand gestures in the frames.

Features

1. Extract frames from a video file at a specified frames per second (FPS).
2. Save the extracted frames as images in a folder with a formatted filename based on the duration of the frame.
3. Utilize the MediaPipe Hand Detection model to detect hand gestures in the saved frames.
4. Print the detected hand gestures and their corresponding labels (e.g., "Right" or "Left") on the console.

Dependencies:
Python 3.x
OpenCV (4.5.4 or higher)
MediaPipe (0.8.13 or higher)

How to run:
1.pip install opencv-python mediapipe
2. Clone or download this repository.
3. Place the video file you want to process in the same directory as the script.
4.Open the script in a Python IDE or text editor and modify the parameters, such as video filename, desired FPS for frame extraction, and hand detection 5. 
5. The script will create a folder with the name of the video file in the same directory, and save the extracted frames as images in that folder.
6. The detected hand gestures and their labels will be printed on the console for each saved frame.

python video_frame_extraction_and_hand_gesture_detection.py

This code is designed to extract frames from a specified video file at a desired frame rate, save them as images, and then detect hand gestures in the saved frames. It is intended for educational and experimental purposes, and may need to be customized for specific use cases or video file formats. Please refer to the documentation of OpenCV and MediaPipe libraries for more information on how to use them effectively. We hope this code is useful for your hand gesture recognition or hand tracking project, and if you have any questions or feedback, please don't hesitate to contact us.

