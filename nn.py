import cPickle
import gzip
import numpy
from random import randint
import os
from PIL import Image
import cv2
nhl = 100

def GetRandomWeights(rows,cols):
	w = numpy.zeros(shape=(rows,cols))
	for i in range(rows):
		for j in range(cols):
			w[i][j] = randint(0,9)/100.0
	return w

def NeuralNetwork(training_data,test_data):
	trainingX = training_data[0]
	trainingY = training_data[1]

	testX = test_data[0]
	testY = test_data[1]

	w1 = GetRandomWeights(nhl,len(trainingX[0]))
	b1 = GetRandomWeights(nhl,1)

	w2 = GetRandomWeights(10,nhl)
	b2 = GetRandomWeights(10,1)

	t = numpy.zeros(shape=(len(trainingX),10))
	for i in range(len(trainingY)):
		t[i][trainingY[i]] = 1

	neta = 0.1

	for l in range(1):
		print("Iteration = %d"%l)
		for i in range(len(trainingX)): 
			
			x = trainingX[i]
			z = numpy.dot(w1,numpy.transpose(x))

			for j in range(nhl):
				z[j] += b1[j]
			z = 1.0/(1.0+numpy.exp(-z))
		
			a = numpy.dot(w2,numpy.transpose(z))

			for j in range(10):
				a[j] += b2[j]

			a = softmax(a)
			dk = a - numpy.transpose(t[i])
			grad2 = Multiply(dk,z)

			temp1 = numpy.dot(numpy.transpose(w2),dk)
			dz = numpy.multiply(z,1-z) * temp1 
			grad1 = Multiply(dz,x)

			w1 = w1 - neta*grad1
			w2 = w2 - neta*grad2

	print ("Results on Training Set")
	TestAccuracy(w1,w2,b1,b2,trainingX,trainingY)
	print ("Results on Test Set")
	TestAccuracy(w1,w2,b1,b2,testX,testY)
	return w1,w2,b1,b2


def TestAccuracy(w1,w2,b1,b2,trainingX,trainingY):

	count = 0
	for i in range(len(trainingX)):
		
		x = trainingX[i]
		z = numpy.dot(w1,numpy.transpose(x))
		for j in range(nhl):
			z[j] += b1[j]

		z = 1/(1+numpy.exp(-z))

		a = numpy.dot(w2,numpy.transpose(z))
		for j in range(10):
				a[j] += b2[j]

		a = softmax(a)

		max = numpy.argmax(a)
		if max == trainingY[i]:
			count += 1


	print ("Total Correct Predictions = %d"%count)
	accuracy = (count/float(len(trainingX))) * 100.0
	print ("Accuracy = %f"%accuracy)




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
print ("\nNeural Network Training \n")

print ("Will Run 5 Iterations, takes around 5-6 mins. Thanks for your patience!")
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

w1,w2,b1,b2 = NeuralNetwork(training_data,test_data)
testDataX,testDataY = GetUSPSData()

print ("Results on USPS Data")
TestAccuracy(w1,w2,b1,b2,testDataX,testDataY)





