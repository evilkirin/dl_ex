import cv2
import numpy as np

img = cv2.imread('imgs/245051.png')
print(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', img)
cv2.waitKey(0)

# _, threshold = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
# print(threshold)

points = cv2.findNonZero(img_gray)
x, y, w, h = cv2.boundingRect(points)
print(x, y, w, h)

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('threshold to roi', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)