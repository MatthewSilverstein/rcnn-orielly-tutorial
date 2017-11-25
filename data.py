from os.path import isfile, join
from os import listdir
import re
import numpy as np

wordListPath = 'resources/wordsList.npy'
wordVectorsPath = 'resources/wordVectors.npy'
positiveFilesPath = 'resources/positiveReviews/'
negativeFilesPath = 'resources/negativeReviews/'
preComputedXImagePath = 'resources/x_image.npy'
preComputedYLabelPath = 'resources/y_label.npy'
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
positiveFiles = [positiveFilesPath + f for f in listdir(positiveFilesPath) if isfile(join(positiveFilesPath, f))]
negativeFiles = [negativeFilesPath + f for f in listdir(negativeFilesPath) if isfile(join(negativeFilesPath, f))]

numFiles = len(positiveFiles) + len(negativeFiles)

wordsList = np.load(wordListPath)
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load(wordVectorsPath)

maxSeqLength = 250
numDimensions = 300

def read_data():
	# preAnalysis() # uncomment to do analysis on the input data
	# Computing the x_image is expensive.
	if not isfile(preComputedXImagePath):
		print('No precomputed x_image found')
		computeXImage()
	if not isfile(preComputedYLabelPath):
		print('No precomputed y_label found')
		computeYLabel()
	x_image = np.load(preComputedXImagePath)
	y_label = np.load(preComputedYLabelPath)
	return x_image, y_label, wordsList, wordVectors, maxSeqLength, numDimensions

def cleanSentences(string):
	string = string.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", string.lower())

def computeXImage():
	x_image = np.zeros((numFiles, maxSeqLength), dtype='int32')
	fileCounter = 0
	for pf in positiveFiles:
		with open(pf, "r") as f:
			indexCounter = 0
			line=f.readline()
			cleanedLine = cleanSentences(line)
			split = cleanedLine.split()
			for word in split:
				try:
					x_image[fileCounter][indexCounter] = wordsList.index(word)
				except ValueError:
					x_image[fileCounter][indexCounter] = 399999 #Vector for unkown words
				indexCounter = indexCounter + 1
				if indexCounter >= maxSeqLength:
					break
			fileCounter = fileCounter + 1 
	for nf in negativeFiles:
		with open(nf, "r") as f:
			indexCounter = 0
			line=f.readline()
			cleanedLine = cleanSentences(line)
			split = cleanedLine.split()
			for word in split:
				try:
					x_image[fileCounter][indexCounter] = wordsList.index(word)
				except ValueError:
					x_image[fileCounter][indexCounter] = 399999 #Vector for unkown words
				indexCounter = indexCounter + 1
				if indexCounter >= maxSeqLength:
					break
			fileCounter = fileCounter + 1
	np.save(preComputedXImagePath, x_image)

def computeYLabel():
	y_label = np.zeros((numFiles, 1), dtype='int32')
	count = 0
	for f in positiveFiles:
		y_label[count][0] = 1
		count += 1
	np.save(preComputedYLabelPath, y_label)

def preAnalysis():
	print(len(wordsList))
	print(wordVectors.shape)

	numWords = []
	for pf in positiveFiles:
	    with open(pf, "r", encoding='utf-8') as f:
	        line=f.readline()
	        counter = len(line.split())
	        numWords.append(counter)       
	print('Positive files finished')

	for nf in negativeFiles:
	    with open(nf, "r", encoding='utf-8') as f:
	        line=f.readline()
	        counter = len(line.split())
	        numWords.append(counter)  
	print('Negative files finished')

	print('The total number of files is', numFiles)
	print('The total number of words in the files is', sum(numWords))
	print('The average number of words in the files is', sum(numWords)/len(numWords))