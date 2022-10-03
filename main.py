import cv2
import numpy as np
import pytesseract
import re
import requests

pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract'
config = ('-l eng --oem 3 --psm 7')

img = cv2.imread('./resource/card.jpg')
g = 3.0

img = img.astype(np.float)
img = ((img / 255) ** (1 / g)) * 255
img = img.astype(np.uint8)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

x, y, w, h = 730, 985, 180, 25
roi = img[y:y+h, x:x+w]
print(roi.shape)
img2 = roi.copy()
cv2.rectangle(roi, (0,0), (w-1, h-1), (0,255,0)) # roi 전체에 사각형 그리기



cv2.imshow("img",img)
cv2.imshow("roi", img2)
roi_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
roi_gray = cv2.resize(roi_gray, dsize=(0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

roi_bi = cv2.threshold(roi_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow("roi_gray", roi_bi)

re_roi_bi = cv2.resize(roi_bi, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.imshow("re", re_roi_bi)

text = pytesseract.image_to_string(re_roi_bi, config=config)
print(text)
text = text.replace(" ", "")
text = text.replace("\n", "")
text_list = text.split('-')
#0 O, l 1, 등 OCR이 구분하지 못하는 경우 치환하며 진행 필요
text_list[0] = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", text_list[0])
text_list[1] = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", text_list[1])
text_list[1] = text_list[1].replace("O","0")
text = text_list[0] + '-' + text_list[1]

print(text)
cardKeyword = text

yugioh_db = "https://www.db.yugioh-card.com/yugiohdb/card_search.action?ope=1&sess=1&rp=20&" +\
            "keyword=" + cardKeyword +\
            "&stype=4&ctype=&othercon=2&starfr=&starto=&pscalefr=&pscaleto=&linkmarkerfr=&linkmarkerto=&link_m=2&atkfr=&atkto=&deffr=&defto=&request_locale=ko"

req = requests.get(yugioh_db)
html = req.text
status = req.status_code
is_ok = req.ok

print(status)
print(req.ok)
print(type(html))

f = open('test.html','w', encoding = 'utf-8')
f.write(html)
f.close()


cv2.waitKey(0)
cv2.destroyAllWindows()