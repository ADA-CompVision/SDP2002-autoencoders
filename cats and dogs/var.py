import os
import numpy as np
import cv2

# Specify the directory containing the images
input_dir = 'new_train'

# Specify the directory where the augmented images will be saved
output_dir = 'outp'

# Define the augmentations
augmentations = [
    ('zoom_in_10', lambda img: cv2.resize(img, None, fx=1.1, fy=1.1)),
    ('zoom_out_10', lambda img: cv2.resize(img, None, fx=0.9, fy=0.9)),
    ('zoom_in_20', lambda img: cv2.resize(img, None, fx=1.2, fy=1.2)),
    ('shift_north', lambda img: cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, -0.1 * img.shape[0]]]), img.shape[:2])),
    ('shift_south', lambda img: cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, 0.1 * img.shape[0]]]), img.shape[:2])),
    ('shift_east', lambda img: cv2.warpAffine(img, np.float32([[1, 0, -0.1 * img.shape[1]], [0, 1, 0]]), img.shape[:2])),
    ('shift_west', lambda img: cv2.warpAffine(img, np.float32([[1, 0, 0.1 * img.shape[1]], [0, 1, 0]]), img.shape[:2])),
    ('shift_northwest', lambda img: cv2.warpAffine(img, np.float32([[1, 0, 0.1 * img.shape[1]], [0, 1, -0.1 * img.shape[0]]]), img.shape[:2])),
    ('shift_northeast', lambda img: cv2.warpAffine(img, np.float32([[1, 0, -0.1 * img.shape[1]], [0, 1, -0.1 * img.shape[0]]]), img.shape[:2])),
    ('shift_southeast', lambda img: cv2.warpAffine(img, np.float32([[1, 0, -0.1 * img.shape[1]], [0, 1, 0.1 * img.shape[0]]]), img.shape[:2])),
    ('shift_southwest', lambda img: cv2.warpAffine(img, np.float32([[1, 0, 0.1 * img.shape[1]], [0, 1, 0.1 * img.shape[0]]]), img.shape[:2])),
    ('horizontal_flip', lambda img: cv2.flip(img, 1))
]

# Loop through the images in the input directory
for filename in os.listdir(input_dir):
    # Read the image
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply the augmentations and save the variations
    for aug_name, aug_func in augmentations:
        aug_image = aug_func(img)
        output_filename = f'{os.path.splitext(filename)[0]}_{aug_name}.jpg'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, aug_image)
