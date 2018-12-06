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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

data_dir = ''			# 紀錄資料路徑
currentfile = ''		# 紀錄目前執行的dir
filelen = 0				# 設置label標記使用
datalist = []			# 紀錄圖片
Xtrain = []
Ytrain = []
unit_num = 0			#輸出神經元個數
predictlist = []
fileorder = []
model = None			# 儲存模型
#curfilelist = []		# 目前儲存的圖片資料集

class FoundException(Exception):
	pass

class NofileException(Exception):
	pass

class NocorrectNum(Exception):
	pass

'''
	執行完目前的dir後,初始化
'''
def clear():
	global data_dir,currentfile,filelen,datalist,Xtrain,Ytrain,unit_num,predictlist,fileorder,model
	data_dir = ''
	currentfile = ''
	datalist.clear()
	filelen = 0
	datalist.clear()
	if type(Xtrain) != type([]) and type(Ytrain) != type([]):
		Xtrain,Ytrain = Xtrain.tolist(),Ytrain.tolist()
	Xtrain.clear()
	Ytrain.clear()
	unit_num = 0
	predictlist.clear()
	fileorder.clear()
	model = None

'''
	尋找路徑中的檔案
'''
def findfile(filename):
	global data_dir,filelen
	data_dir = os.path.join('data',filename)
	if not(os.path.exists(data_dir)):
		raise FoundException()
	else:
		filelist = os.listdir(data_dir)
		if len(filelist)!=0:
			for file in filelist:
				if not(file == 'model'):
					print('findfile:',os.path.join(data_dir,file))     #測試路徑值
					if os.path.isdir(os.path.join(data_dir,file)):	#判斷是否為資料夾目錄	
						readData(file)
		else:
			raise NofileException()

	#print('findfile:',os.listdir(data_dir))   # 以list儲存 目錄下所有文件or目錄
				
'''
	讀取圖片並轉為0~255圖片資訊
'''
def readData(filename):
	global datalist,data_dir,currentfile
	print('data_dir',data_dir)
	datalist.clear()
	currentfile = filename
	data_pic = os.path.join(data_dir,filename)
	print('readData:',data_pic)
	files = os.listdir(data_pic)  #返回目錄下所有檔案file(這裡用於找尋圖片檔)
	#print(type(files)) 			# class:list
	print('filelist:')
	if len(files)!=0:
		for file in files:
			print(os.path.join(data_pic,file))   #測試取得路徑是否正確
			datapath = os.path.join(data_pic,file)
			img = cv2.imread(datapath,0)		 # (arg1=路徑,arg2=灰階讀取),讀取圖片以nparray方式儲存
			img = cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)		#縮為32x32大小圖片(統一規格)
			#print(img.shape)
			npimg = np.array(img)
			npimg = npimg.flatten()				# 轉為一維array
			datalist.append(npimg)				# 累積增加至list中,直到最後圖檔完成
	else:
		raise NofileException()	
	#print('datalist_type',type(datalist))
	np_data = np.array(datalist)			# 轉換nparray
	print('npimg:',npimg.shape)				#測試nparray轉換是否正確
	packingData(np_data.tolist())			# 再轉成list格式

'''
	利用pickle把資料壓縮於一個檔案
'''
def packingData(data):
	global filelen,fileorder
	print('data:',np.array(data).shape)
	print('datalen:',len(data))
	print('file:',currentfile)
	print('datadir:',data_dir)
	labels = [filelen]*len(data)
	fileorder.append(currentfile)
	#curfilelist = fileorder
	dict = {'filename':"%s"%currentfile,'data':data,'labels':labels}
	with open(data_dir+"\\data_%s"%currentfile, 'wb') as f:
		pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)
	filelen+=1

def one_hot(x,n):
	x = np.array(x)
	print('x:',x)
	#print('eye:',type(np.eye(n)))
	return np.eye(n)[x]

'''
	載入pickle檔案
'''
def loadData(n):
	global Xtrain,Ytrain,unit_num
	unit_num = n
	pkfiles = os.listdir(data_dir)
	print('filelen:',filelen)
	if filelen== n:
		for file in pkfiles:
			if os.path.isfile(os.path.join(data_dir,file)):
				batch = np.load(os.path.join(data_dir,file))
				print('pickleData:',batch.keys())
				print('labels:',batch['labels'])
				data = np.array(batch['data'])/255.0
				print('data:',len(data))
				Xtrain.append(np.array(data).astype('float32'))
				OH_label = one_hot(batch['labels'],n)
				print('label:',OH_label)
				Ytrain.append(np.array(OH_label).astype('float32'))
		Xtrain = np.concatenate(Xtrain, axis=0)
		print('datas:',np.array(Xtrain).shape,'type:',type(Xtrain))
		Ytrain = np.concatenate(Ytrain,axis=0)
		print('labels:',np.array(Ytrain).shape)
	else:
		raise NocorrectNum()

def inputpic(filepath):
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
	
'''
	建立模型部分
'''
def modelcreate():
	global unit_num,Xtrain,Ytrain,model
	model = Sequential()  #建立模型
	model.add(Dense(units=256,input_dim=1024,kernel_initializer='normal',activation='relu'))
 	# 這行units需要配合分類種數設置
	model.add(Dense(units=unit_num,kernel_initializer='normal',activation='softmax')) 

	print('model:')
	print(model.summary())

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	train_history=model.fit(x=Xtrain,y=Ytrain,validation_split=0.0,epochs=20,batch_size=2,verbose=2)
	scores = model.evaluate(Xtrain,Ytrain)

	print('scores:',scores)

'''
	模型預測部分
'''
def modelpredict(path):
	global predictlist
	# 設定輸入圖片路徑 input('圖片路徑')
	predicts = model.predict(inputpic(path))
	#print('predicts:',test_predict)
	print('one_predict:',predicts.tolist()[0],'type:',type(predicts.tolist()))

	predictlist = predicts.tolist()

'''
	這邊會自動產生訓練好的模型,使用model_reuse.py載入即可使用
'''
def savemodel():
	global data_dir,model
	modelfile = 'model'
	if not(os.path.exists(os.path.join(data_dir,modelfile))):
		os.mkdir(os.path.join(data_dir,modelfile))
		model.save(os.path.join(os.path.join(data_dir,modelfile),'model.h5'))
	else:
		print('data_dir',data_dir)
		model.save(os.path.join(os.path.join(data_dir,modelfile),'model.h5'))

'''
  這邊會判斷結果, 下面請依照圖片集的資料順序下去設定,因為這邊設計是按照順序依序讀入圖片集
  ex. 索引值從0開始, 0->代表判斷為第一個圖片集分類, 1-> 代表判斷為第二個圖片集分類, 以此分類...
'''
def answer(n):
	global predictlist,fileorder
	print('fileorder:',fileorder)
	for select in range(0,n):
		if predictlist[0][select]==1.0:
			print("this is a %s"%fileorder[select])
'''
def getfile():
	return curfilelist
'''

'''
	輸入部分
	findfile('輸圖片集資料夾路徑')
	loadData('輸入分類種數')
'''

while (1):
	clear()
	try:
		arg1 = input("請輸入圖片集資料夾的路徑:")
		arg2 = int(input("請輸入分類數目:"))
		arg3 = input("請輸入要預測的圖片路徑:")
		if type(arg1)==type('str'):
			print('arg1:',arg1,',arg2:',arg2,',arg3:',arg3)
			findfile(arg1)
			loadData(int(arg2))
			modelcreate()
			savemodel()
			modelpredict(arg3)
			answer(int(arg2))
		else:
			print('輸入錯誤,請重新再次輸入:')
	except ValueError as ve:
		print('error:請輸入數字')
	except FoundException as fe:
		print('error:file doesn\'t exist')
	except NofileException as nf:
		print('error:no set of picture in this file.')
	except NocorrectNum as nc:
		print('classificative num does not correct.')
	except Exception:
		print('error:請重新再次輸入')
	
