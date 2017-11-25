from os.path import isfile, join
from os import listdir
import re
import numpy as np
from tensorflow.python.framework import random_seed

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

	
	train_data = DataSet(
		np.concatenate((x_image[0:11499],	x_image[13499:24999]), axis=0), 
		np.concatenate((y_label[0:11499],	y_label[13499:24999]), axis=0))
	test_data = DataSet(x_image[11499:13499], y_label[11499:13499])
	data = Data(train_data, test_data)
	return data, wordsList, wordVectors, maxSeqLength, numDimensions

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
	y_label = np.zeros((numFiles, 2), dtype='int32')
	count = 0
	for f in positiveFiles:
		y_label[count] = [0, 1]
		count += 1
	for f in negativeFiles:
		y_label[count] = [1, 0]
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

class Data:
	def __init__(self, train_data, test_data):
		self.train = train_data
		self.test_data = test_data

class DataSet(object):

	def __init__(self, images, labels, seed=None):
		seed1, seed2 = random_seed.get_seed(seed)
		# If op level seed is not set, use whatever graph level seed is returned
		np.random.seed(seed1 if seed is None else seed2)
		self._num_examples = images.shape[0]
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False, shuffle=True):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		# Shuffle for the first epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self._num_examples)
			np.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]
		# Go to the next epoch
		if start + batch_size > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self._num_examples - start
			images_rest_part = self._images[start:self._num_examples]
			labels_rest_part = self._labels[start:self._num_examples]
			# Shuffle the data
			if shuffle:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._images = self.images[perm]
				self._labels = self.labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size - rest_num_examples
			end = self._index_in_epoch
			images_new_part = self._images[start:end]
			labels_new_part = self._labels[start:end]
			return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._images[start:end], self._labels[start:end]