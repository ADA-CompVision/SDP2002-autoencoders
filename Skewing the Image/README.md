Description

This code provides examples of how to perform image transformation techniques on an input image using OpenCV and scikit-image libraries. The transformations demonstrated include skewing, shifting, and scaling of an image.

Dependencies

The following libraries are required to run the code:
OpenCV (version X.X.X or higher)
scikit-image (version X.X.X or higher)
numpy (version X.X.X or higher)

Installation
Install the required dependencies using the following commands:

pip install opencv-python
pip install scikit-image
pip install numpy

Functions
The code will perform the following transformations on the input image:
Skewing: It uses the skimage.transform.AffineTransform function from scikit-image to apply a shear transformation on the image.
Shifting: It uses the cv2.warpAffine function from OpenCV to apply a shift transformation on the image.
Scaling: It uses the cv2.resize function from OpenCV to resize the image with a scaling factor.
Results
The transformed images will be saved as JPEG files in the same directory with the following filenames:
skewed.jpeg: the skewed image
shifted.jpg: the shifted image
scaled_image.jpeg: the scaled image
