import numpy as np
import cv2

img = cv2.imread("digits.png")

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
outerBox = cv2.adaptiveThreshold(
    img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
outerBox = cv2.bitwise_not(outerBox)

kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], np.uint8)

outerBox = cv2.dilate(outerBox, kernel)

contours, hierarchy = cv2.findContours(
    image=outerBox, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area > 175:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        img = cv2.drawContours(image=img, contours=contour, contourIdx=-1,
                               color=(50, 100, 150), thickness=1, lineType=cv2.LINE_AA)

cv2.imwrite("mydigits.png", img)
