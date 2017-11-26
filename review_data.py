from os.path import isfile, join
from os import listdir
import re
import numpy as np
import data


default_positive_files_path = 'resources/positiveReviews/'
default_negative_files_path = 'resources/negativeReviews/'
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

default_max_seq_length = 250
default_num_dimensions = 300
processed_images_path = 'resources/processed_images.npy'
processed_labels_path = 'resources/processed_labels.npy'

class ReviewProcessor:
	def __init__(
		self,
		words_list,
		word_vectors,
		positive_files_path=default_positive_files_path,
		negative_files_path=default_negative_files_path,
		max_seq_length=default_max_seq_length,
		num_dimensions=default_num_dimensions):
		
		self.words_list = words_list
		self.word_vectors = word_vectors
		self.positive_files_path = positive_files_path
		self.negative_files_path = negative_files_path
		self.max_seq_length = max_seq_length
		self.num_dimensions = num_dimensions

	def process_data(self):
		self.read_reviews()
		#self.analyze_reviews() # Uncomment to analyze reviews
		self.process_reviews()
		self.build_datasets()

	def read_reviews(self):
		self.positive_files = [self.positive_files_path + f for f in listdir(self.positive_files_path) if isfile(join(self.positive_files_path, f))]
		self.negative_files = [self.negative_files_path + f for f in listdir(self.negative_files_path) if isfile(join(self.negative_files_path, f))]
		self.num_files = len(self.positive_files) + len(self.negative_files)

	def analyze_reviews(self):
		print(len(self.words_list))
		print(len(self.word_vectors.shape))

		num_words = []
		for pf in self.positive_files:
			with open(pf, "r", encoding='utf-8') as f:
				line=f.readline()
				counter = len(line.split())
				num_words.append(counter)		 
		print('Positive files finished')

		for nf in self.negative_files:
			with open(nf, "r", encoding='utf-8') as f:
				line=f.readline()
				counter = len(line.split())
				numWords.append(counter)	
		print('Negative files finished')

		print('The total number of files is', self.num_files)
		print('The total number of words in the files is', sum(num_words))
		print('The average number of words in the files is', sum(num_words)/len(num_words))

	def process_reviews(self):
		if not isfile(processed_images_path):
			print('No processed images found, processing images')
			self.process_images()
			print('Finished processing images')
		if not isfile(processed_labels_path):
			print('No processed labels found, processing labels')
			self.process_labels()
			print('Finished processing labels')
		self.images = np.load(processed_images_path)
		self.labels = np.load(processed_labels_path)

	def process_images(self):
		images = np.zeros((self.num_files, self.max_seq_length), dtype='int32')
		file_counter = 0
		for pf in self.positive_files:
			with open(pf, "r") as f:
				index_counter = 0
				line=f.readline()
				cleaned_line = self.clean_sentences(line)
				split = cleaned_line.split()
				for word in split:
					try:
						images[file_counter][index_counter] = self.words_list.index(word)
					except ValueError:
						images[file_counter][index_counter] = 399999 #Vector for unkown words
					index_counter += 1
					if index_counter >= self.max_seq_length:
						break
				file_counter += 1 
		for nf in self.negative_files:
			with open(nf, "r") as f:
				index_counter = 0
				line=f.readline()
				cleaned_line = self.clean_sentences(line)
				split = cleaned_line.split()
				for word in split:
					try:
						images[file_counter][index_counter] = self.words_list.index(word)
					except ValueError:
						images[file_counter][index_counter] = 399999 #Vector for unkown words
					index_counter += 1
					if index_counter >= self.max_seq_length:
						break
				file_counter += 1
		np.save(processed_images_path, images)

	def clean_sentences(self, string):
		string = string.lower().replace("<br />", " ")
		return re.sub(strip_special_chars, "", string.lower())

	def process_labels(self):
		labels = np.zeros((self.num_files, 2), dtype='int32')
		count = 0
		for f in self.positive_files:
			labels[count] = [0, 1]
			count += 1
		for f in self.negative_files:
			labels[count] = [1, 0]
			count += 1
		np.save(processed_labels_path, labels)

	def build_datasets(self):
		train_data = data.DataSet(
			np.concatenate((self.images[0:11499], self.images[13499:24999]), axis=0), 
			np.concatenate((self.labels[0:11499], self.labels[13499:24999]), axis=0))
		test_data = data.DataSet(self.images[11499:13499], self.labels[11499:13499])
		self.data = data.Data(train_data, test_data)