file = '/home/mohammad/Desktop/Fig0944(a)(calculator).tif'


import numpy as np
import cv2


def show(*args):
    for idx, arg in enumerate(args):
        cv2.namedWindow(str(idx), cv2.NORMAL_CLONE)
        cv2.moveWindow(str(idx), 350*idx, 10)
        cv2.imshow(str(idx), arg)
        cv2.imwrite(str(idx)+'.png', arg)

    cv2.waitKey()

org = cv2.imread(file, 0)#[5:212, 5:255]
org = cv2.medianBlur(org,5)
th3 = cv2.adaptiveThreshold(org, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)

ret, th1 = cv2.threshold(org, 150, 255, cv2.THRESH_BINARY)


mgh = cv2.morphologyEx(th1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
mgh = cv2.dilate(
    mgh, cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6)))

contours, hierarchy = cv2.findContours(mgh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
br = [cv2.boundingRect(cnt) for cnt in contours]
br = filter(lambda x: x[2] > 15 and x[3] > 18, br)
br = sorted(br, key=lambda x: x[2], reverse=True)
for idx, r in enumerate(br):
    x,y,w,h = r
    if h > 18 and w > 15:
        cv2.putText(org,str(idx),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.rectangle(org,(x,y),(x+w,y+h),(0,255,0),2)

show(org)#, th3, th1, mgh)#, horizontal, without_hor_lines, vertical, without_ver_lines)
cv2.destroyAllWindows()

