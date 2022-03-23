# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:08:42 2022

@author: BERKAY_ZGN
"""
import cv2 as cv,numpy as np
# from operator import itemgetter, attrgetter

def clockwise_order(contour):
    cx, cy = int(contour[:, 0, 0].mean()), int(contour[:, 0, 1].mean())
    x = contour - [cx,cy]
    b= 0
    freelist= [0,1,2,3]
    while b < len(contour):

        if x[b][0][0] < 0 and x[b][0][1] < 0:
                freelist[0] = contour[b]
        elif x[b][0][0] < 0 and x[b][0][1] > 0:
                freelist[1] = contour[b]
        elif x[b][0][0] > 0 and x[b][0][1] > 0:
                freelist[2] = contour[b]
        elif x[b][0][0] > 0 and x[b][0][1] < 0:
                freelist[3] = contour[b]
        b += 1             
    return freelist

def l2norm(pt1,pt2,pt3,pt4):
    w1 = np.sqrt(((pt1[0] -pt2[0] ) ** 2) + ((pt1[1] -pt2[1]) ** 2))
    w2 = np.sqrt(((pt3[0] -pt4[0] ) ** 2) + ((pt3[1] -pt4[1] ) ** 2))
    maxw = max(int(w1),int(w2))
    h1 = np.sqrt(((pt1[0] -pt4[0] ) ** 2) + ((pt1[1] -pt4[1]) ** 2))
    h2 = np.sqrt(((pt3[0] -pt2[0] ) ** 2) + ((pt3[1] -pt2[1] ) ** 2))
    maxh = max(int(h1),int(h2))
    return maxw,maxh

img = cv.imread("arkaa.jpg")
blur = cv.GaussianBlur(img,(3,3),0)
g_img = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
bifi = cv.bilateralFilter(g_img,8,250,250)
ret,th = cv.threshold(bifi,150,255,cv.THRESH_BINARY)
canny = cv.Canny(th,170,255)
cont,_ = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# cv.imshow("th",th)
# cv.imshow("can",canny)
# cv.imshow("bifi", bifi)
cv.waitKey(0)
cv.destroyAllWindows()
if len(cont) > 0:
    max_contour = max(cont, key=cv.contourArea)
    epsilon = 0.01 * cv.arcLength(max_contour, True)
    avecont = cv.approxPolyDP(max_contour,epsilon, True,)
    y = clockwise_order(avecont)
else:
    print("couldnt find any contours")

mw, mh = l2norm(avecont[1][0],avecont[0][0],avecont[3][0],avecont[2][0])
coor_in = np.float32(y)
coor_out = np.float32([[0,0],[0,mh-1],[mw-1,mh-1],[mw-1,0]])
M = cv.getPerspectiveTransform(coor_in,coor_out)
out = cv.warpPerspective(img,M,(mw, mh),flags=cv.INTER_LINEAR)

cv.imshow("gor",out)
cv.imshow("kagit",img)
cv.waitKey(0)    