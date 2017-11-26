import numpy as np
from os.path import isfile, join
import tensorflow as tf
import datetime
from random import randint
import review_data

def getTrainBatch():
	return dataset.train.next_batch(batchSize)

def getTestBatch():
	return dataset.test.next_batch(batchSize)

wordListPath = 'resources/wordsList.npy'
wordVectorsPath = 'resources/wordVectors.npy'
wordsList = np.load(wordListPath)
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load(wordVectorsPath)


batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
maxSeqLength = 250
numDimensions = 300


tf.reset_default_graph()

review_processor = review_data.ReviewProcessor(wordsList, wordVectors, max_seq_length=maxSeqLength, num_dimensions=numDimensions)
review_processor.process_data()
dataset = review_processor.data

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