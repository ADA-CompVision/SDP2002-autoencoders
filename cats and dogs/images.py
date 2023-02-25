import os
import shutil
import cv2

# Set the source and destination directories
# src_dir = 'outp'
# dst_dir = 'new_train'

# # Get a list of all files in the source directory
# files = os.listdir(dst_dir)

# # Sort the files alphabetically to ensure consistent results
# print(len(files))
# # count=0
# # #Get the first 695 files and move them to the destination directory
# # for file in files:
# #     src_path = os.path.join(src_dir, file)
# #     dst_path = os.path.join(dst_dir, file)
# #     shutil.move(src_path, dst_path)

img = cv2.imread('new_train\cat.106_horizontal_flip.jpg')

# Get the shape of the image
height, width, channels = img.shape

# Print the shape of the image
print(f"Image shape: {width}x{height}x{channels}")