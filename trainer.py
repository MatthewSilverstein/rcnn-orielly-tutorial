import numpy as np
from os.path import isfile, join
import tensorflow as tf
import datetime
from random import randint
import review_data

class Trainer:

	def __init__(
		self,
		datasets,
		word_vectors,
		max_seq_length=250,
		batch_size=24,
		output_keep_prob=0.75,
		lstm_units=1,
		num_dimensions=1

		):

		self.datasets = datasets
		self.batch_size=batch_size
		num_classes = datasets.train.labels.shape[1]

		tf.reset_default_graph()

		labels = tf.placeholder(tf.float32, [batch_size, num_classes])
		input_data = tf.placeholder(tf.int32, [batch_size, max_seq_length])
		self.labels = labels
		self.input_data = input_data

		data = tf.nn.embedding_lookup(word_vectors, input_data)

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
		lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=output_keep_prob)

		rnn, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
		weight = tf.Variable(tf.truncated_normal([lstm_units, num_classes]))
		bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
		value = tf.transpose(rnn, [1, 0, 2])
		last = tf.gather(value, int(value.get_shape()[0]) - 1)
		self.prediction = (tf.matmul(last, weight) + bias)

		self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=labels))
		self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

	def train(self, iterations):
		sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

		tf.summary.scalar('Loss', self.loss)
		tf.summary.scalar('Accuracy', self.accuracy)
		merged = tf.summary.merge_all()
		logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
		writer = tf.summary.FileWriter(logdir, sess.graph)

		for i in range(iterations):
			next_batch, next_batch_labels = self.datasets.train.next_batch(self.batch_size)
			sess.run(self.optimizer, {self.input_data: next_batch, self.labels: next_batch_labels})
			#Write summary to Tensorboard
			if (i % 50 == 0):
				summary = sess.run(merged, {self.input_data: next_batch, self.labels: next_batch_labels})
				writer.add_summary(summary, i)
			#Save the network every 10,000 training iterations
			if (i % 10000 == 0 and i != 0):
				save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
				print("saved to %s" % save_path)
		writer.close()
