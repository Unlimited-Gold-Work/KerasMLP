'''
	本程式為MLP Keras 圖形辨識
	著作: 馬太
	就讀學校: 國立臺北教育大學
	壓縮檔內有許多自行寫得api文本,尚未理解者請先去自行查看,若真不會或有BUG出現,
	請於github搜尋Unlimited-Gold-Work,並自行留言

	This program is for testing MLP.
	Author: Martai
	University: National Taipei University of Education
	There are lots of API documents in compressed file by myself,user can first
	check it. If you don't really understand or find bugs in my program, please
	search the keyword named 'Unlimited-Gold-Work' on github and leave a message
	on messege board.  	 
'''

import cv2
import os
import pickle
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Activation

data_dir = 'data'
predictlist = []
fileorder = []
model = None

class FoundException(Exception):
	pass

'''
	歸零化
'''
def clear():
	global data_dir
	data_dir = 'data'
	predictlist.clear()
	fileorder.clear()

'''
	輸入圖片路徑
'''
def inputpic(filepath):
	try:
		if os.path.exists(filepath):		
			img = cv2.imread(filepath,0)
			img = cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
			npimg = np.array(img)
			npimg = npimg.flatten()
			npimg = np.expand_dims(npimg,axis=0)
			print('input:',npimg.shape)
			return npimg
		else:
			raise FoundException()
	except FoundException as fe:
		print('error:file doesn\'t exist')

'''
	輸入圖片總集的最上層路徑,即data資料夾底下
'''
def datapath(filename):
	global data_dir
	data_dir = os.path.join(data_dir,filename)
	if not(os.path.exists(data_dir)):
		raise FoundException()
	else:
		data_pic = os.path.join(data_dir,'model')
		return data_pic

'''
	載入模組用
'''
def loadmodel(dir):
	global model
	model = load_model(os.path.join(datapath(arg1),'model.h5'))
	print(model.summary())

def loadpickle():
	global data_dir
	print(os.listdir(data_dir))
	pkfiles = os.listdir(data_dir)
	for file in pkfiles:
		if os.path.isfile(os.path.join(data_dir,file)):
			batch = np.load(os.path.join(data_dir,file))
			fileorder.append(batch['filename'])

'''
	預測圖片
'''
def predictpic(dir):
	global predictlist,model
	# 與main.py一樣, 輸入要判斷的圖檔路徑
	predicts = model.predict(inputpic(dir))
	predictlist = predicts.tolist()
	print('one_predict:',predicts.tolist()[0],'type:',type(predicts.tolist()))
'''
	與main.py一樣,依照資料夾順序進行設定,0 -> 第一類圖片集, 1->第二類圖片集
'''
def answer():
	global fileorder,predictlist
	for select in range(0,len(fileorder)):
		if predictlist[0][select]==1.0:
			print('this is a %s'%fileorder[select])


while (1):
	try:
		clear()
		arg1 = input("請輸入圖片集資料夾路徑:")
		arg2 = input("請輸入要預測的圖片路徑:")
		loadmodel(arg1)
		loadpickle()
		predictpic(arg2)
		answer()
	except FoundException as fe:
		print('error:file doesn\'t exist')
	except Exception:
		print('輸入錯誤,請重新輸入')
