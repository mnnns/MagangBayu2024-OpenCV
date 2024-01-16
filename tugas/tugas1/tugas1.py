import cv2
import numpy as np

lower_color = np.array([80, 20, 20])
upper_color = np.array([100, 255, 255])

### buka/jalankan code ini dari file MagangBayu2024-OpenCV (sesuaikan direktori jika perlu)
image_path = 'tugas/tugas1/tugas1.png'
img = cv2.imread(image_path)

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) != 0:
    for contour in contours:
        if cv2.contourArea(contour) > 600:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

cv2.imshow("Color Mask", color_mask)
cv2.imshow("Annotated Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
