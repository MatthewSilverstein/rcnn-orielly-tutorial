import numpy as np
from os import listdir
from os.path import isfile, join
import re
import tensorflow as tf
import datetime

wordListPath = 'resources/wordsList.npy'
wordVectorsPath = 'resources/wordVectors.npy'
positiveFilesPath = 'resources/positiveReviews/'
negativeFilesPath = 'resources/negativeReviews/'
preComputedIdsMatrixPath = 'resources/idsMatrix.npy'

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
def cleanSentences(string):
	string = string.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", string.lower())

def computeIdsMatrix():
	ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
	fileCounter = 0
	for pf in positiveFiles:
		with open(pf, "r") as f:
			indexCounter = 0
			line=f.readline()
			cleanedLine = cleanSentences(line)
			split = cleanedLine.split()
			for word in split:
				try:
					ids[fileCounter][indexCounter] = wordsList.index(word)
				except ValueError:
					ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
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
					ids[fileCounter][indexCounter] = wordsList.index(word)
				except ValueError:
					ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
				indexCounter = indexCounter + 1
				if indexCounter >= maxSeqLength:
					break
			fileCounter = fileCounter + 1 
	np.save(preComputedIdsMatrixPath, ids)

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

wordsList = np.load(wordListPath)
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load(wordVectorsPath)
print ('Loaded the word vectors!')

print(len(wordsList))
print(wordVectors.shape)

positiveFiles = [positiveFilesPath + f for f in listdir(positiveFilesPath) if isfile(join(positiveFilesPath, f))]
negativeFiles = [negativeFilesPath + f for f in listdir(negativeFilesPath) if isfile(join(negativeFilesPath, f))]
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

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))

maxSeqLength = 250
numDimensions = 300

# Computing the idsMatrix is expensive.
if not isfile(preComputedIdsMatrixPath):
	print('No precomputed idsMatrixFound')
	computeIdsMatrix()

ids = np.load(preComputedIdsMatrixPath)
print('Loaded precomputed idsMatrix')


batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)




sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(iterations):
#Next Batch of reviews
	nextBatch, nextBatchLabels = getTrainBatch();
	sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
	#Write summary to Tensorboard
	if (i % 50 == 0):
		summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
		writer.add_summary(summary, i)
	#Save the network every 10,000 training iterations
	if (i % 10000 == 0 and i != 0):
		save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
		print("saved to %s" % save_path)
writer.close()