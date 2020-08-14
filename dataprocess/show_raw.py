import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.fromfile(r'/mnt/02520c27-ec8e-4661-b88f-05aa2011ffa7/lhk/47_HD_1.raw', dtype='float32')
img = img.reshape((512,512))
cv2.imshow('', img)
cv2.waitKey()
cv2.destroyWindow()
plt.imshow(img, cmap=plt.cm.gray)
plt.show()