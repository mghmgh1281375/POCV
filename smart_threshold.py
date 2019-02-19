
import cv2
import numpy as np

c = 1
gamma_list = [0.1, 0.2, 0.5, 1., 5., 10.]
image = cv2.imread('./resources/photo_2019-02-19_21-19-39.jpg', 0)

# Show original image
cv2.imshow('original', image)

# Normalizing image
normalized = image / 255

for gamma in gamma_list:

    # thresholding
    thresh = c * np.power(image, gamma)
    cv2.imshow('gamma-{:.3}'.format(gamma), thresh)
    cv2.waitKey()

cv2.destroyAllWindows()
