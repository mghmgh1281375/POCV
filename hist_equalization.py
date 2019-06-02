# editing from uni
# Histogram equalizatio

import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('resources/photo_2019-02-19_21-19-39.jpg', 0)
hist, bins = np.histogram(image.flatten(), 256, [0, 256])
# plt.plot(hist, c='b')
# plt.show()
normalized = hist / (image.shape[0]*image.shape[1])
# plt.plot(normalized, c='r')
# plt.show()

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(image.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img2 = cdf[image]

cv2.imshow('mine', img2)
# # cv2.imwrite(s, 'resources/photo_histogram_quantization.png')


img3 = cv2.equalizeHist(image)
cv2.imshow('cv2', img3)


cv2.waitKey()
cv2.destroyAllWindows()


equalized = np.cumsum(normalized)

plt.plot(equalized, label='equalized', alpha=0.5)
plt.plot(normalized, label='hist', alpha=0.5)
plt.legend(loc='upper center')
plt.show()

