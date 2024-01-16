import cv2
import numpy as np

def process_image(img_path):
    img = cv2.imread(img_path)
    cropped_img = img[200:500, 650:900]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img, (5, 5), 300)

    canny = cv2.Canny(img_blur, 125, 175)
    edges = cv2.Canny(gray, 90, 130)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    morph = cv2.morphologyEx(edges, cv2.RETR_TREE, kernel)

    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    contour_img = cropped_img.copy()
    cv2.drawContours(cropped_img, contours, 0, (0, 0, 0), 4)

    peri = cv2.arcLength(big_contour, True)
    approx = cv2.approxPolyDP(big_contour, 0.01 * peri, True)

    img = cv2.putText(img, str(len(approx)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

    print("Jumlah sisi: ", len(approx))
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ### buka/jalankan code ini dari direktori MagangBayu2024-OpenCV (ubah direktori dibawah jika perlu)
    image_path = 'tugas/tugas2/tugas2.jpg'
    process_image(image_path)