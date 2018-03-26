# !/usr/bin/python
# -*- coding:utf-8 -*-
from Tkinter import *
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import tkMessageBox
from morphing import *
import cv2
import os

def resourcePath(relativePath):
	try:
		basePath = sys._MEIPASS
	except Exception:
		basePath = os.path.abspath(".")
	return os.path.join(basePath, relativePath)

def selectImage(panelStr):
	# grab a reference to the image panels
	global panelA, panelB, img1, img2

	if panelStr != "panelA" and panelStr != "panelB":
		print("return")
		return
	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkFileDialog.askopenfilename()

	# ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		img = cv2.imread(path)
		showImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		showImg = Image.fromarray(showImg)

		# ...and then to ImageTk format
		showImg = ImageTk.PhotoImage(showImg)

		# if the panels are None, initialize them
		#if panelA is None or panelB is None:
		if panelStr == "panelA":
			img1 = img
			panelA.configure(image=showImg)
			panelA.image = showImg
		elif panelStr == "panelB":
			img2 = img
			panelB.configure(image=showImg)
			panelB.image = showImg
		else:
			pass

def selectModel():
	global modelSrc
	path = tkFileDialog.askopenfilename()
	modelSrc = path

def selectDir():
	global fileDir
	path = tkFileDialog.askdirectory()
	fileDir = path

# initialize the window toolkit along with the two image panels
root = Tk()
modelSrc = None
fileDir = None
root.title("人脸变换 in Python - 3150103960 边嘉蒙")
placeholder = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.imread(resourcePath("placeholder.jpeg")), cv2.COLOR_BGR2RGB)))
panelA = Label(image=placeholder)
panelA.pack(side="left", padx=10, pady=10)
panelB = Label(image=placeholder)
panelB.pack(side="right", padx=10, pady=10)
img1 = None
img2 = None
label = Label(root, text='中间帧数: 1 到 60')
label.pack()
spv = StringVar()
spinbox = Spinbox(root, from_=1, to=60, textvariable = spv)
spinbox.pack()
btn4 = Button(root, text="选择训练模型", command=selectModel)
btn4.pack()

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI

btn1 = Button(root, text="←载入第 1 张图像", command=lambda:selectImage("panelA"))
btn1.pack(side="bottom", expand="yes", padx="10", pady="10")
btn2 = Button(root, text="载入第 2 张图像→", command=lambda:selectImage("panelB"))
btn2.pack(side="bottom", expand="yes", padx="10", pady="10")
btn3 = Button(root, text="生成中间帧与变化录像", command=lambda:process(img1, img2, int(spv.get()), modelSrc, fileDir))
btn3.pack(side="bottom", expand="yes", padx="10", pady="10")
btn5 = Button(root, text="选择存放生成文件的文件夹", command=lambda:selectDir())
btn5.pack(side="bottom", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()