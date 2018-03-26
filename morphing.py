# !/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import dlib
import string
import os
import numpy as np
import tkMessageBox

# 调用 dlib 获取特征点
def getLandmark(model, img, kptList, subdiv):
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = model
    faces = detector(img, 1)
    if (len(faces) > 0):
        for k,d in enumerate(faces):
            #cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
            shape = landmark_predictor(img,d)
            for i in range(68):
                # 顺便存储三角分割
                subdiv.insert((shape.part(i).x, shape.part(i).y))
                kptList.append((shape.part(i).x, shape.part(i).y))
                #cv2.circle(img, (shape.part(i).x, shape.part(i).y),5,(0,255,0), -1, 8)
                #cv2.putText(img,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,2555,255))
    #cv2.imshow('Frame',img)
    # 为了后面可以更好地变换，手工加八个点（照片的四个顶点和四个中点）
    width = img.shape[1]-1
    height = img.shape[0]-1
    subdiv.insert((0, 0))
    subdiv.insert((0, height))
    subdiv.insert((width, 0))
    subdiv.insert((width, height))
    subdiv.insert((0, height/2))
    subdiv.insert((width/2, height))
    subdiv.insert((width, height/2))
    subdiv.insert((width/2, 0))
    kptList.append((0, 0))
    kptList.append((0, height))
    kptList.append((width, 0))
    kptList.append((width, height))
    kptList.append((0, height/2))
    kptList.append((width/2, height))
    kptList.append((width, height/2))
    kptList.append((width/2, 0))

    return

# If point is in the rect
def isInRect(rect, point):
    if point[0]<rect[0]:
        return False
    elif point[1]<rect[1]:
        return False
    elif point[0]>rect[2]:
        return False
    elif point[1]>rect[3]:
        return False
    return True

# 获得三角剖分的对应顶点
def getDelaunay(rect, subdiv, kptList):
    dTriIndex = []
    triangleList = subdiv.getTriangleList()
    for t in triangleList:
        point1 = (t[0], t[1])
        point2 = (t[2], t[3])
        point3 = (t[4], t[5])

        if(isInRect(rect, point1) and isInRect(rect, point2) and isInRect(rect, point3)):
            count = 0
            triVertex = [-1, -1, -1]
            # 三角形的顶点是特征点下标，计算对应的三角形顶点并且存储
            for k in range(0, 76):
                if(abs(point1[0]-kptList[k][0])<1.0 and abs(point1[1]-kptList[k][1])<1.0):
                    triVertex[0] = k
                    count += 1
                if(abs(point2[0]-kptList[k][0])<1.0 and abs(point2[1]-kptList[k][1])<1.0):
                    triVertex[1] = k
                    count += 1
                if(abs(point3[0]-kptList[k][0])<1.0 and abs(point3[1]-kptList[k][1])<1.0):
                    triVertex[2] = k
                    count += 1
            if count == 3:
                dTriIndex.append(triVertex)
    return dTriIndex

# 绘制计算得到的 Delaunay 三角形
def drawDelaunay(img, kptList, dTriIndex):
    lineColor = (0, 0, 255)
    for p in dTriIndex:
        cv2.line(img, kptList[p[0]], kptList[p[1]], lineColor, 1)
        cv2.line(img, kptList[p[1]], kptList[p[2]], lineColor, 1)
        cv2.line(img, kptList[p[2]] , kptList[p[0]], lineColor, 1)
    return

# 对三角形仿射变换
def morphTriangle(img1, img2, dTriIndex1, dTriIndex2, dTriIndex, kptList1, kptList2, kptList, alpha):

    triV1 = []
    triV2 = []
    triV = []
    triV_int = []

    # 遍历三个 dTriIndex 中的点，把三角形顶点坐标从 (特征点下标, 特征点下标) 改为对应的 (x, y) 
    for p in dTriIndex:
        triV.append(((kptList[p[0]][0], kptList[p[0]][1]), (kptList[p[1]][0], kptList[p[1]][1]), (kptList[p[2]][0], kptList[p[2]][1])))
        triV1.append(((kptList1[p[0]][0], kptList1[p[0]][1]), (kptList1[p[1]][0], kptList1[p[1]][1]), (kptList1[p[2]][0], kptList1[p[2]][1])))
        triV2.append(((kptList2[p[0]][0], kptList2[p[0]][1]), (kptList2[p[1]][0], kptList2[p[1]][1]), (kptList2[p[2]][0], kptList2[p[2]][1])))
    triV1 = np.array(np.float32(triV1))
    triV2 = np.array(np.float32(triV2))
    triV = np.array(np.float32(triV))

    img1Dst = np.zeros((img1.shape), dtype=np.uint8)
    img2Dst = np.zeros((img2.shape), dtype=np.uint8)

    for i in range(0, triV.shape[0]):
        mask1 = np.zeros((img1.shape), dtype=np.uint8)
        mask2 = np.zeros((img2.shape), dtype=np.uint8)
        warpMat1 = cv2.getAffineTransform(triV1[i], triV[i])
        warpMat2 = cv2.getAffineTransform(triV2[i], triV[i])
        roiCorners1 = np.array(triV1[i], dtype=np.int32)
        roiCorners2 = np.array(triV2[i], dtype=np.int32)
        channelCount = img1.shape[2]
        ignoreMaskColor = (255,)*channelCount
        cv2.fillConvexPoly(mask1, roiCorners1, ignoreMaskColor)
        cv2.fillConvexPoly(mask2, roiCorners2, ignoreMaskColor)
        img1Warp = cv2.bitwise_and(img1, mask1)
        img2Warp = cv2.bitwise_and(img2, mask2)
        img1Warp = cv2.warpAffine(img1Warp, warpMat1, (img1Warp.shape[1], img1Warp.shape[0]))
        img2Warp = cv2.warpAffine(img2Warp, warpMat2, (img2Warp.shape[1], img2Warp.shape[0]))

        (B1, G1, R1) = cv2.split(img1Warp)
        (B2, G2, R2) = cv2.split(img2Warp)

        rect1 = cv2.boundingRect(B1)
        rect2 = cv2.boundingRect(B2)
        for j in range(rect1[1], rect1[1]+rect1[3]):
            for k in range(rect1[0], rect1[0]+rect1[2]):
                if(any(img1Warp[j, k, :]) and (img1Warp[j, k, :]>img1Dst[j, k, :]).any()):
                    img1Dst[j, k, :] = img1Warp[j, k, :]
        for j in range(rect2[1], rect2[1]+rect2[3]):
            for k in range(rect2[0], rect2[0]+rect2[2]):
                if(any(img2Warp[j, k, :]) and (img2Warp[j, k, :]>img2Dst[j, k, :]).any()):
                    img2Dst[j, k, :] = img2Warp[j, k, :]

    return img1Dst, img2Dst

def doAlpha(img1Dst, img2Dst, alpha):
    imgWarpDst = np.zeros((img1Dst.shape), dtype=np.uint8)
    imgWarpDst = cv2.addWeighted(img1Dst, 1-alpha, img2Dst, alpha, 0)

    return imgWarpDst

def process(img1, img2, frame, modelsrc, fileDir):
    if img1 is None or img2 is None:
        tkMessageBox.showerror("出错啦", "两张图片都要载入哦")
        return
    if frame < 1 or frame > 60:
        tkMessageBox.showerror("出错啦", "中间帧数最少是 1 ，最多是 60 ")
        return
    if img1.shape != img2.shape:
        tkMessageBox.showerror("出错啦", "两张图片的长、宽、通道数必须一致")
        return
    if modelsrc is None:
        tkMessageBox.showerror("出错啦", "请载入模型")
    if fileDir is None:
        tkMessageBox.showerror("出错啦", "请选择存放路径")
    print(fileDir)
    print("start running")

    # 分别检测两张人脸的特征点，并且存储+计算三角剖分
    model = dlib.shape_predictor(modelsrc)
    
    size = img1.shape
    rect1 = (0, 0, size[1], size[0])
    subdiv1 = cv2.Subdiv2D(rect1)
    kptList1 = []

    rect2 = (0, 0, size[1], size[0])
    subdiv2 = cv2.Subdiv2D(rect2)
    kptList2 = []
    
    getLandmark(model, img1, kptList1, subdiv1)
    getLandmark(model, img2, kptList2, subdiv2)

    dTriIndex1 = getDelaunay(rect1, subdiv1, kptList1)
    dTriIndex2 = getDelaunay(rect2, subdiv2, kptList2)

    #drawDelaunay(img1, kptList1, dTriIndex1)
    #drawDelaunay(img2, kptList2, dTriIndex2)
    #cv2.imshow("Delaunayimg1", img1)
    #cv2.imshow("Delaunayimg2", img2)

    tempx = 0.0
    tempy = 0.0
    imgWarpDstList = []


    for i in range(0, frame+2):
        alpha = float(i)/(frame+1)
        print(alpha)
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect)
        kptList = []
        for i in range(0, 76):
            tempx = (1-alpha)*kptList1[i][0]+alpha*kptList2[i][0]
            tempy = (1-alpha)*kptList1[i][1]+alpha*kptList2[i][1]
            kptList.append((tempx, tempy))
            subdiv.insert(kptList[i])
        dTriIndex = getDelaunay(rect, subdiv, kptList)
        ## 前后两帧的变形
        (img1Dst, img2Dst) = morphTriangle(img1, img2, dTriIndex1, dTriIndex2, dTriIndex, kptList1, kptList2, kptList, alpha)
        imgWarpDstList.append(doAlpha(img1Dst, img2Dst, alpha))

    video = cv2.VideoWriter(fileDir+"/"+"ResultVideo.mp4", cv2.VideoWriter_fourcc(*'XVID'), 24.0, (size[1], size[0]))
    for i in range(0, frame+2):
        video.write(imgWarpDstList[i])
    
    #imageio.mimwrite('result.gif', imgWarpDstList, 'GIF', duration=0.1)
    for i in range(0, frame+2):
       cv2.imwrite(fileDir+"/"+"imgWarpDstList"+str(i)+'.jpg', imgWarpDstList[i])
    #    cv2.imshow("imgWarpDstList"+str(i), imgWarpDstList[i])
    tkMessageBox.showinfo("完成啦", "请在目录下查看结果")
