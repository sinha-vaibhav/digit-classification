import cPickle
import gzip
import numpy
from random import randint
import os
from PIL import Image
import cv2

def GetRandomWeights():
	w = numpy.zeros(shape=(10,784))
	for i in range(10):
		for j in range(784):
			w[i][j] = randint(0,9)/10.0
	return w

def GradientDescent(training_data,test_data):
	w = GetRandomWeights()
	winit = w
	b = numpy.ones(shape=(10,1))

	trainingX = training_data[0]
	trainingY = training_data[1]

	t = numpy.zeros(shape=(50000,10))
	for i in range(len(trainingY)):
		t[i][trainingY[i]] = 1

	
	neta = 0.01
	for l in range(15):
		print("Iteration = %d"%l)

		for i in range(50000):

			x = trainingX[i]
			y = numpy.dot(w,numpy.transpose(x))
		
			for j in range(10):
				y[j] += b[j]


			y = softmax(y)
			

			diff = y - numpy.transpose(t[i])
			grad = Multiply(diff,x)
			w = w - neta*grad
	print ("======================")
	
	
	count = 0
	for i in range(50000):
		y = numpy.dot(w,numpy.transpose(trainingX[i]))
		for j in range(10):
			y = numpy.add(y,b[j])
		y = softmax(y)
		max = numpy.argmax(y)
		if max == trainingY[i]:
			count += 1


	print ("Total Correct Predictions on Training = %d"%count)
	accuracy = (count/50000.0) * 100.0
	print ("Training Accuracy = %f"%accuracy)

	testX = test_data[0]
	testY = test_data[1]
	
	count = 0
	for i in range(len(testX)):
		y = numpy.dot(w,numpy.transpose(testX[i]))
		for j in range(10):
			y = numpy.add(y,b[j])
		y = softmax(y)
		max = numpy.argmax(y)
		if max == testY[i]:
			count += 1


	print ("Total Correct Predictions on Test = %d"%count)
	accuracy = (count/float(len(testX))) * 100.0
	print ("Test Accuracy = %f"%accuracy)

	return w,b

def TestAccuracy(w,b,testX,testY):
	count = 0
	for i in range(len(testX)):
		y = numpy.dot(w,numpy.transpose(testX[i]))
		for j in range(10):
			y = numpy.add(y,b[j])
		y = softmax(y)
		max = numpy.argmax(y)
		if max == testY[i]:
			count += 1


	print ("Total Correct Predictions on USPS Data = %d"%count)
	accuracy = (count/float(len(testX))) * 100.0
	print ("Accuracy on USPS Data = %f"%accuracy)

	

def Multiply(diff,x):


	diff = numpy.asmatrix(diff)
	x = numpy.asmatrix(x)
	res = numpy.transpose(diff) * x
	
	res = numpy.array(res)

	return res

def softmax(y):
	
	y = numpy.exp(y)
	total = numpy.sum(y)
	y = numpy.divide(y,total)

	return numpy.transpose(y)

def GetUSPSData():
	curPath = os.path.dirname(os.path.abspath(__file__))
	curPath += '/USPSData/Numerals'
	testDataX = []
	testDataY = []

	for i in range(9):
		curFolderPath = curPath + '/' + str(i)
		imgs =  os.listdir(curFolderPath)
		for img in imgs:
			curImg = curFolderPath + '/' + img
			if curImg[-3:] == 'png':
				curImg = cv2.imread(curImg,0)
				curImg = cv2.resize(curImg,(28,28))
				curImg = numpy.array(curImg)
				curImg = 255 - curImg
				curImg = curImg.ravel()
				curImg = curImg/255.0

				testDataX.append(curImg)
				testDataY.append(i)

	return testDataX,testDataY


print ("Name : Vaibhav Sinha")
print ("UB Person # : 50208769")
print ("UD ID : vsinha2")
print ("\nLogistic Regression \n")

print ("Will Run 15 Iterations, takes around 3-4 mins. Thanks for your patience!")


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()
testDataX,testDataY = GetUSPSData()
w,b = GradientDescent(training_data,test_data)

TestAccuracy(w,b,testDataX,testDataY)


