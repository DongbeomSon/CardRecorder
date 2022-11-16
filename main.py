import cv2
import imutils as imutils
import numpy as np
import pytesseract
import re
import requests #request html
from bs4 import BeautifulSoup #html parsing
from imutils.contours import sort_contours
from imutils.perspective import four_point_transform
import pandas as pd
import openpyxl

def findRec(org_img):
    img = imutils.resize(org_img, 500)
    ratio = org_img.shape[1] / float(img.shape[1])

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blurred, 100, 255, L2gradient=True)

    cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = []
    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 사각형만 판별해서 저장
        if len(approx) == 4:
            findCnt.append(approx)

    # 만약 추출한 윤곽이 없을 경우 오류
    if len(findCnt) == 0:
        raise Exception(("인식 실패"))

    cards = []
    for c in findCnt:
        cards.append(four_point_transform(org_img, c.reshape(4, 2) * ratio))

    return cards

org_img = cv2.imread('./resource/card16.jpg')
cards = findRec(org_img)
i = 0
for idx, img in enumerate(cards):

    cv2.imshow("img" + str(idx),img)

cv2.waitKey(0)
cv2.destroyAllWindows()