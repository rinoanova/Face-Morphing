# Face-Deformation
Final Project for Computer Animation in ZJU 2017

[TOC]

## 实现功能

- 输入两张照片，输出从第一张渐变到第二张的录像，可以指定帧数

## 运行环境

- Python 2.7
- 需要[ dlib 人脸训练模型](https://github.com/davisking/dlib)

## 实现思路

1. 用 dlib 训练模型识别人脸特征点 + 手工指定图片四角+四边中点为特征点
2. 对人脸进行 Delaunay 三角剖分
3. 利用特征点数组线性插值 + 仿射变换生成中间帧图像，合成视频

## Reference

- [基于 openCV + Dlib 的面部合成](https://blog.csdn.net/wangxing233/article/details/51549880), hahaha233, CSDN
- [Image Warping and Morphing](http://graphics.cs.cmu.edu/courses/15-463/2011_fall/Lectures/morphing.pdf), Alexei Efros, CMU
- [Delaunay Triangulation and Voronoi Diagram using OpenCV](https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/), Satya Mallick, LearnOpenCV
