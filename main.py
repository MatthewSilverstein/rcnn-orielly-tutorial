import numpy as np
from os.path import isfile, join
import tensorflow as tf
import datetime
from random import randint
import review_data
from trainer import Trainer

def main():
	word_list_path = 'resources/wordsList.npy'
	word_vectors_path = 'resources/wordVectors.npy'
	words_list = np.load(word_list_path)
	words_list = words_list.tolist() #Originally loaded as numpy array
	words_list = [word.decode('UTF-8') for word in words_list] #Encode words as UTF-8
	word_vectors = np.load(word_vectors_path)

	batch_size = 24
	lstm_units = 64
	num_classes = 2
	iterations = 100000
	max_seq_length = 250
	num_dimensions = 300
	output_keep_prob = 0.75

	review_processor = review_data.ReviewProcessor(words_list, word_vectors, max_seq_length=max_seq_length, num_dimensions=num_dimensions)
	review_processor.process_data()
	datasets = review_processor.data

	trainer = Trainer(
		datasets, 
		word_vectors,
		batch_size=24,
		lstm_units=lstm_units,
		num_dimensions=300,
		max_seq_length=250,
		output_keep_prob=0.75)
	trainer.train(iterations)


if __name__ == "__main__":
	main()

