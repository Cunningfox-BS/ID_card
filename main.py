from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#将图片显示以及销毁集合为一个函数cv_show（）
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_contours(cnts,method="left-to-right"):
    reverse=False
    i=0
    # 选择水平或者竖直
    if method=="right-to-left" or method=="bottom-to-top":
        reverse=True
    if method=="top-to-bottom" or method=="bottom-to-top":
        i=1
    #会形成一个包含xywh的元组为元素的列表
    boundingBoxes=[cv2.boundingRect(c) for c in cnts]

    #cnts和boundingBoxes通过zip函数绑定，key=lambda x:x[1][i]中x代表的是前面zip(cnts,boundingBoxes)表示的内容
    (cnts,boundingBoxes)=zip(*sorted(zip(cnts,boundingBoxes),key=lambda x:x[1][i],reverse=reverse))
    return cnts,boundingBoxes


#根据读入的width等比缩放hight
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


#读入模板并且进行预处理
img=cv2.imread("template.png")#读入
ref=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰度图
ref=cv2.threshold(ref,127,255,cv2.THRESH_BINARY_INV)[1]#二值图


refCnts,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#计算轮廓cv2.RETR_EXTERNAL（外边框）cv2.CHAIN_APPROX_SIMPLE（终点坐标）
refCnts=sort_contours(refCnts,method="left-to-right")[0]#排序，从左到右
digits={}

for (i,c) in enumerate(refCnts):
    (x,y,w,h)=cv2.boundingRect(c)
    roi=ref[y:y+h,x:x+w]
    roi=cv2.resize(roi,(57,88))
    #计算外接矩形并且resize成合适大小
    digits[i]=roi#每一个i都代表一片用于检测的区域


#设置卷积核np.ones（）
rectKernel=np.ones((3,9),np.uint8)
sqKernel=np.ones((5,5),np.uint8)

#读入预处理
image=cv2.imread("card_02.png")
image=resize(image,width=300)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#礼帽操作突出明亮区域，数字相较于背景较为明亮cv2.morphologyEx（）
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)

#计算梯度以计算轮廓cv2.Sobel（）
#这几部数值转换是因为cv2.Sobel中的ddepth=cv2.CV_32F，不一定是0-255，图片深度不同
gradX=cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradX=np.absolute(gradX)
(minVal,maxVal)=(np.min(gradX),np.max(gradX))
gradX=(255*((gradX-minVal)/(maxVal-minVal)))
gradX=gradX.astype("uint8")


#通过闭操作把数字连在一起cv2.morphologyEx（）
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)

#cv2.THRESH_OTSU会选择合适的阈值
thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

#闭操作再一次cv2.morphologyEx（）
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,sqKernel)

#计算轮廓cv2.findContours（）
threshCnts,hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
locs=[]

#遍历轮廓
for (i,c) in enumerate(cnts):
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)
    # 选择合适的区域
    if ar>2.5 and ar<4.0:
        if (w>40 and w<55) and (h>10 and h<20):
            locs.append((x,y,w,h))

#符合的轮廓从左到右排序sorted（）
locs=sorted(locs,key=lambda x:x[0])
output=[]

#遍历每一个轮廓中的数字
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    #根据坐标提取每一个组
    groupOutput=[]
    group=gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    group=cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    #计算轮廓cv2.findContours（）
    digitCnts,hierarchy=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #计算每一组内的（即单个数字）轮廓
    # 根据坐标大小从左到右排序并且选取sort_contours（）第一个返回值
    digitCnts=sort_contours(digitCnts,method="left-to-right")[0]

    #计算每一组中的每一个数值
    for c in digitCnts:
        #找到当前数值的轮廓，resize成合适的的大小cv2.boundingRect（）
        (x,y,w,h)=cv2.boundingRect(c)
        roi=group[y:y+h,x:x+w]
        roi=cv2.resize(roi,(57,88))
        # cv_show('roi', roi)
        #计算匹配得分
        scores=[]
        #在模板中计算得分
        for (digit,digitROI) in digits.items():
            #模板匹配cv2.matchTemplate（）
            result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_)=cv2.minMaxLoc(result)
            scores.append(score)

        #找到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

        #画出来cv2.rectangle（）
        cv2.rectangle(image, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 得到结果
        output.extend(groupOutput)

#打印结果
image=cv2.resize(image,(0,0,),fx=2.0,fy=2.0)
cv2.imshow("Image", image)
cv2.waitKey(0)

