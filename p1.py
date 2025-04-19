import os

import cv2


# read image
image_path = os.path.join('.', 'cake.jpg')

img = cv2.imread(image_path)

# write image

cv2.imwrite(os.path.join('.', 'cake_out.jpg'), img)

# visualize image

cv2.imshow('image', img)
cv2.waitKey(0)