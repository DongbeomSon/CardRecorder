import cv2
import imutils as imutils
import numpy as np
import pytesseract
import re
import requests #request html
from bs4 import BeautifulSoup #html parsing
from imutils.contours import sort_contours
from imutils.perspective import four_point_transform


def adjust_gamma(img, g):
    img = img.astype(np.float64)
    img = ((img / 255) ** (1 / g)) * 255
    img = img.astype(np.uint8)

    return img

def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def findRec(org_img):
    img = imutils.resize(org_img, 500)
    ratio = org_img.shape[1] / float(img.shape[1])

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blurred, 100, 255, L2gradient=True)

    cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None
    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
        if len(approx) == 4:
            findCnt = approx
            break

    # 만약 추출한 윤곽이 없을 경우 오류
    if findCnt is None:
        raise Exception(("Could not find outline."))

    output = img.copy()

    cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

    card = four_point_transform(org_img, findCnt.reshape(4, 2) * ratio)
    return card

pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'
config = ('-l eng --oem 3 --psm 7')

org_img = cv2.imread('./resource/card11.jpg')
#org_img = adjust_gamma(org_img,2.5)
cv2.imshow("org",org_img)
card = findRec(org_img)
cv2.imshow("jaebal",card)

h,w = card.shape[:2]
gray = grayscale(card)


roi = card[int(h*0.71):int(h*0.75), int(w*0.71):int(w*0.91)]
cv2.imshow("roi",roi)
roi = cv2.resize(roi, dsize=(0, 0), fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
cv2.imshow("roi2",roi)
roi = grayscale(roi)
gray = roi.copy()

#roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,0)
roi = cv2.GaussianBlur(roi, (5, 5), 0)
roi = cv2.threshold(roi,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow("otsu",roi)

###
#ray = cv2.cvtColor(roi_for, cv2.COLOR_BGR2GRAY)
(H, W) = gray.shape

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 21))

gray = cv2.GaussianBlur(gray, (11, 11), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")

grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

close_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
close_thresh = cv2.erode(close_thresh, None, iterations=2)

cv2.imshow("hi",close_thresh)

cnts = cv2.findContours(close_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="top-to-bottom")[0]

roi_list = []
roi_title_list = []

margin = 10
(x, y, w, h) = cv2.boundingRect(cnts[0])

#roi = cv2.rectangle(gray, (x - margin, y - margin), (x + w + margin, y + h + margin), (0,255,0), 2)
roi = gray[y-margin:y+h+margin, x-margin:x+w+margin]
cv2.imshow("hi2",roi)

roi = cv2.GaussianBlur(roi, (5, 5), 0)
roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57,0)
roi = thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
text = pytesseract.image_to_string(roi)
cv2.imshow("real",roi)

print(text)
print("hi..")

###
def loc(event,x,y,flags,param):
    if(event == 1):
        print(x/w)
        print(y/h)
cv2.setMouseCallback("jaebal",loc)

cv2.waitKey(0)
cv2.destroyAllWindows()