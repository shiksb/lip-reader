
import numpy as np
import os
from keras.models import Sequential
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers.wrappers import TimeDistributed
from keras.applications.vgg16 import VGG16
from keras.models import Model
import ssl
import imageio


# configures the paramaters of the neural net
class Config(object):
	def __init__(self, nc, ne, msl, bs, lr, dp):
		self.CLASSES = nc
		self.EPOCHS = ne
		self.MAX_SEQ = msl
		self.BATCH_SIZE= bs
		self.LR = lr
		self.MAX_WIDTH = 74
		self.MAX_HEIGHT = 74
		self.dp = dp


# main class
class LipReader(object):
	def __init__(self, config):
		self.config = config		


	# this method yields the training data
	def training_generator(self):
		while True:
			for i in range(int(np.shape(self.X_train)[0] / self.config.BATCH_SIZE)):
				x = self.X_train[i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE]
				y = self.y_train[i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE]
				one_hot_labels_train = keras.utils.to_categorical(y, num_classes=self.config.CLASSES)
				yield (x,one_hot_labels_train)


	# this method creates and saves the model
	def create_model(self, seen_validation):
		np.random.seed(0)

		bottleneck_train_path = 'bottleneck_features_train.npy'
		bottleneck_val_path = 'bottleneck_features_val.npy'
		top_model_weights = 'bottleneck_TOP_LAYER.h5'
		
		if seen_validation is False:
			top_model_weights = 'unseen_bottleneck_TOP_LAYER.h5'
			bottleneck_train_path = 'unseen_bottleneck_features_train.npy'
			bottleneck_val_path = 'unseen_bottleneck_features_val.npy'

		ssl._create_default_https_context = ssl._create_unverified_context

		input_layer = keras.layers.Input(
			shape=(self.config.MAX_SEQ, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))

		# VGG Layer
		# vgg_base = VGG16(weights='imagenet', include_top=False,
		# 				 input_shape=(self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
        #
		# vgg = Model(input=vgg_base.input, output=vgg_base.output)
        #
		# for layer in vgg.layers[:15]:
		# 	layer.trainable = False
        #
		# x = TimeDistributed(vgg)(input_layer)
        #
		# model = Model(input=input_layer, output=x)
        #
		# x = keras.layers.core.Dropout(rate=self.config.dp)(x)
		# x = keras.layers.core.Dense(10)(x)
		# p = keras.layers.core.Activation('softmax')(x)
        #
		# bottleneck_model = Model(input=input_layer, output=p)
        #
		# print('Creating bottleneck features...')
		# if not os.path.exists(bottleneck_train_path):
		# 	bottleneck_features_train = bottleneck_model.predict_generator(self.training_generator(),
		# 	steps=np.shape(self.X_train)[0] / self.config.BATCH_SIZE)
		# 	np.save(bottleneck_train_path, bottleneck_features_train)
		# print('Created bottleneck train')
		# if not os.path.exists(bottleneck_val_path):
		# 	bottleneck_features_val = bottleneck_model.predict(self.X_val)
		# 	np.save(bottleneck_val_path, bottleneck_features_val)
		# print('Created bottleneck val')
		# bottleneck_model.save_weights(top_model_weights)
		# print('Saved bottleneck weights')
		################################################################################################################


		adam = keras.optimizers.SGD(lr=self.config.LR) #, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

		one_hot_labels_train = keras.utils.to_categorical(self.y_train, num_classes=self.config.CLASSES)
		one_hot_labels_val =   keras.utils.to_categorical(self.y_val, num_classes=self.config.CLASSES)

		train_data = np.load(bottleneck_train_path)
		val_data =   np.load(bottleneck_val_path)
		print(np.shape(self.X_train))
		print(np.shape(self.X_val))

		print(len(train_data), len(one_hot_labels_train))
		print(len(val_data),   len(one_hot_labels_val))


		# Deep Convolutional Layer
		conv2d1 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
		x = TimeDistributed(conv2d1)(input_layer) #input_shape=(self.config.MAX_SEQ, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3)
		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
		x = keras.layers.core.Activation('relu')(x)
		x = keras.layers.core.Dropout(rate=self.config.dp)(x)
		pool1 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
		x = TimeDistributed(pool1)(x)

		conv2d2 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
		x = TimeDistributed(conv2d2)(x)
		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
		x = keras.layers.core.Activation('relu')(x)
		x = keras.layers.core.Dropout(rate=self.config.dp)(x)
		pool2 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
		x = TimeDistributed(pool2)(x)

		conv2d3 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
		x = TimeDistributed(conv2d3)(x)
		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
		x = keras.layers.core.Activation('relu')(x)
		x = keras.layers.core.Dropout(rate=self.config.dp)(x)
		pool3 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
		x = TimeDistributed(pool3)(x)

		model = Model(input=input_layer, output=x)
		################################################################################################################



		# LSTM Layer
		input_layer_2 = keras.layers.Input(shape=model.output_shape[1:])

		x = TimeDistributed(keras.layers.core.Flatten())(input_layer_2)
		lstm = keras.layers.recurrent.LSTM(256)
		x = keras.layers.wrappers.Bidirectional(lstm, merge_mode='concat', weights=None)(x)

		x = keras.layers.core.Dropout(rate=self.config.dp)(x)
		x = keras.layers.core.Dense(10)(x)
		preds = keras.layers.core.Activation('softmax')(x)

		model_top = Model(input=input_layer_2, output=preds)

		x = model(input_layer)
		preds = model_top(x)

		final_model = Model(input=input_layer, output=preds)
		################################################################################################################



		final_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


		print('Fitting the model...')
		history = final_model.fit_generator(self.training_generator(), steps_per_epoch=np.shape(self.X_train)[0] / self.config.BATCH_SIZE,\
					 epochs=self.config.EPOCHS, validation_data=(self.X_val, one_hot_labels_val))
		
		self.create_plots(history)


		print('Evaluating the model...')
		score = model.evaluate(self.X_val, one_hot_labels_val, batch_size=self.config.BATCH_SIZE)

		print('Finished training, with the following val score:')
		print(score)


	'''
	def create_minibatches(self, data, shape):
		data = [self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test]
		for dataset in 	
			batches = []
			for i in range(0, len(data), self.config.BATCH_SIZE)
				sample = data[i:i + self.config.BATCH_SIZE]
				if len(sample) < self.config.BATCH_SIZE:
					pad = np.zeros(shape)
					sample.extend(pad * (size - len(sample)))
				batches.append(sample)
	'''

	def create_plots(self, history):
		os.mkdir('plots')
		# summarize history for accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig('plots/acc_plot.png')
		plt.clf()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig('plots/loss_plot.png')



	def load_data(self, seen_validation):
		data_dir = 'data'
		if seen_validation:
			data_dir = 'data_seen'

		if os.path.exists('../' + data_dir):
			print('loading saved data...')
			self.X_train = np.load('../' +  data_dir + '/X_train.npy')
			self.y_train = np.load('../'+ data_dir +'/y_train.npy')

			self.X_val = np.load('../'+ data_dir +'/X_val.npy')
			self.y_val = np.load('../'+data_dir+'/y_val.npy')

			self.X_test = np.load('../'+data_dir+'/X_test.npy')
			self.y_test = np.load('../'+data_dir+'/y_test.npy')
			print('Read data arrays from disk.npy')
		else:
			people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
			data_types = ['words']
			folder_enum = ['01','02','03','04','05','06','07','08','09','10']

			UNSEEN_VALIDATION_SPLIT = ['F05']
			UNSEEN_TEST_SPLIT = ['F06']

			SEEN_VALIDATION_SPLIT = ['02']
			SEEN_TEST_SPLIT = ['01']

			self.X_train = []
			self.y_train = []

			self.X_val = []
			self.y_val = []

			self.X_test = []
			self.y_test = [] 

			directory = 'cropped'
			for person_id in people:
				for data_type in data_types: 
					for word_index, word in enumerate(folder_enum):
						for iteration in folder_enum:
							path = os.path.join(directory, person_id, 'words', word, iteration)
							filelist = os.listdir(path + '/')
							sequence = []
							for img_name in filelist:
								if img_name.startswith('color'):
									image = imageio.imread(path + '/' + img_name)
									image = image[:self.config.MAX_WIDTH,:self.config.MAX_HEIGHT,...]
									if image.shape != (74,74,3):
										print('INVALID DIMENSIONS')
									else:
										sequence.append(image)
							pad_array = [np.zeros((self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))]
							sequence.extend(pad_array * (self.config.MAX_SEQ - len(sequence)))
							sequence = np.stack(sequence, axis=0)
							
							if seen_validation == False:
								if person_id in UNSEEN_TEST_SPLIT:
									self.X_test.append(sequence)
									self.y_test.append(word_index)
								elif person_id in UNSEEN_VALIDATION_SPLIT:
									self.X_val.append(sequence)
									self.y_val.append(word_index)
								else:
									self.X_train.append(sequence)
									self.y_train.append(word_index)
							else:
								if iteration in SEEN_TEST_SPLIT:
									self.X_test.append(sequence)
									self.y_test.append(word_index)
								elif iteration in SEEN_VALIDATION_SPLIT:
									self.X_val.append(sequence)
									self.y_val.append(word_index)
								else:
									self.X_train.append(sequence)
									self.y_train.append(word_index)

				print('Finished reading images for person ' + person_id)
			
			print('Finished reading images.')
			print(np.shape(self.X_train))
			self.X_train = np.stack(self.X_train, axis=0)	
			self.X_val = np.stack(self.X_val, axis=0)
			self.X_test = np.stack(self.X_test, axis=0)
			print('Finished stacking the data into the right dimensions. About to start saving to disk...')		
			os.mkdir('../' + data_dir)
			np.save('../'+data_dir+'/X_train', self.X_train)
			np.save('../'+data_dir+'/y_train', np.array(self.y_train))
			np.save('../'+data_dir+'/X_val', self.X_val)
			np.save('../'+data_dir+'/y_val', np.array(self.y_val))
			np.save('../'+data_dir+'/X_test', self.X_test)
			np.save('../'+data_dir+'/y_test', np.array(self.y_test))
			print('Finished saving all data to disk.')

		print('X_train shape: ', np.shape(self.X_train))
		print('y_train shape: ', np.shape(self.y_train))

		print('X_val shape: ', np.shape(self.X_val))
		print('y_val shape: ', np.shape(self.y_val))

		print('X_test shape: ', np.shape(self.X_test))
		print('y_test shape: ', np.shape(self.y_test))



if __name__ == '__main__':
	num_epochs = 10
	learning_rates = 0.0001
	batch_size = 50
	dropout = 0.2

	config = Config(10, num_epochs, 22, batch_size, learning_rates, dropout)
	lipReader = LipReader(config)
	lipReader.load_data(False)
	lipReader.create_model(False)
