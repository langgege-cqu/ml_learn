import numpy as np
import cv2 as cv

img = cv.imread("12.jpg", cv.IMREAD_COLOR)
cv.imshow('lenna', img)
cv.waitKey(0)
print(img.shape)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
print(gray.shape)
cv.waitKey(0)
cv.destroyAllWindows()