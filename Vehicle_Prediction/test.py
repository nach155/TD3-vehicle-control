import cv2
import numpy as np

WINDOW_SIZE = 600
image = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3))
cv2.fillConvexPoly(image, np.array([[-500,500],[550,550],[610,550],[600,500]]), color=(255,0,255))
cv2.imshow('image',image)
cv2.waitKey(0)