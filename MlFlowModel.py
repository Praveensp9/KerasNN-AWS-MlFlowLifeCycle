import os
from mlflow import log_metric, log_param, log_artifact
import mlflow
import mlflow.sklearn
from keras.datasets import imdb
import numpy as np
from keras.optimizers import SGD
from keras import models
import keras as keras
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import sys
import warnings
import argparse
import boto3
import botocore

# AWS S3 bucket name where Kubernetes cluster data is stored
BUCKET_NAME = <Your_S3_BUCKET_NAME>

# Calculates the Precision, Recall , F1 Score after the end of all epochs
# Returns the metrics dictionary
class Metrics(Callback):
	def on_train_begin(self, logs={}):
	 self.metrics = {}
	 
	def on_epoch_end(self, epoch, logs={}):
	 val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
	 val_targ = self.validation_data[1]
	 f1score = f1_score(val_targ, val_predict)
	 recall = recall_score(val_targ, val_predict)
	 precision = precision_score(val_targ, val_predict)
	 self.metrics['recall'] = recall
	 self.metrics['precision'] = precision
	 self.metrics['f1'] = f1score

	def getMetrics(self):
		return self.metrics

# Downloading and Loading the imdb dataset for Keras NN Model
class LoadData:
	def __init__(self):
		print("Inside constructor")
		self.train = []
		self.labels = []

	def load(self):
		(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=30000)
		self.train = np.concatenate((train_data, test_data), axis=0)
		self.labels = np.concatenate((train_labels, test_labels), axis=0)
		return self.train,self.labels

	def normalizeData(self,data,dim=30000):
		results = np.zeros((len(data), dim))
		for i, j in enumerate(data):
			results[i, j] = 1
		return results

	def prepareData(self,data,labels):
		test_X = data[:10000]
		test_Y = labels[:10000]
		train_X = data[10000:]
		train_Y = labels[10000:]
		return train_X,train_Y,test_X,test_Y;

# Preparing the Keras Neural network model which has 3 dense layers
class prepareKerasModel:
	def __init__(self):
		print("inside Keras Model")
		self.model = ''

	def createModel(self,learning_rate=0.01):
		self.model = models.Sequential()

		# Input - Layer , Activation function : relu
		self.model.add(layers.Dense(50, activation = "relu", input_shape=(30000,)))

		# Hidden - Layers , Activation function : relu
		self.model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
		self.model.add(layers.Dense(50, activation = "relu"))
		self.model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
		self.model.add(layers.Dense(50, activation = "relu"))

		# Output- Layer , Activation function : sigmoid
		self.model.add(layers.Dense(1, activation = "sigmoid"))
		self.model.summary()

		# compiling the model
		sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer = sgd,loss = "binary_crossentropy",metrics = ["accuracy"])

# main function
if __name__ == "__main__":

	# My AWS instance where Kubernetes cluster is hosted
	TRACKING_URI = <YOUR-TRACKING-URI>
	mlflow.set_tracking_uri(TRACKING_URI)
	client = mlflow.tracking.MlflowClient(TRACKING_URI)
	warnings.filterwarnings("ignore")
		
	if len(sys.argv) < 3:
		print("Please pass the 3 HyperParameters as command line arguments with the format epoch,batch_size,new")
		exit(0);

	# Passing 3 Hyperparameters as command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch')
	parser.add_argument('--batchsize')
	parser.add_argument('--learningrate')
	argss = parser.parse_args()

	epoch = int(argss.epoch)         #HyperParameter 1, => P1
	batch_size = int(argss.batchsize)    #HyperParameter 1, => P1
	learning_rate = float(argss.learningrate)    #HyperParameter 1, => P3

	# Loading the data
	loaddata = LoadData();
	data,labels = loaddata.load();

	# Normalizing the data
	data = loaddata.normalizeData(data)
	labels = np.array(labels).astype("float32")

	# splitting the data into train and test set
	trainX,trainY,testX,testY = loaddata.prepareData(data,labels);

	# preparing the Keras Model
	model = prepareKerasModel()
	model.createModel(learning_rate)

	# Mlflow logging the 3 hyperparameters and the metrics associated (Accuracy, Precision,Recall, F1-Score)
	# mlflow.set_experiment("Experiment1")
	with mlflow.start_run():
		# Training keras Model and Testing on the validation data
		metrics = Metrics()
		results = model.model.fit(trainX, trainY,epochs= epoch,batch_size = batch_size,validation_data = (testX, testY),callbacks=[metrics])
		kerasmetrics = metrics.getMetrics()

		# mlflow.set_tag("Low Learning rate","low learning_rate ")
		# Mlflow logging the Hyperparameters
		mlflow.log_param("epochs", epoch)									# Epochs
		mlflow.log_param("batch_size", batch_size)							# batch size
		mlflow.log_param("learning_rate", learning_rate)					# learning rate

		# Mlflow logging the Metrics
		mlflow.log_metric("Accuracy", np.mean(results.history["val_acc"]))  # Accuracy
		mlflow.log_metric("Precision", np.mean(kerasmetrics['recall']))		# Precision
		mlflow.log_metric("Recall", np.mean(kerasmetrics['precision']))		# Recall
		mlflow.log_metric("F1 Score", np.mean(kerasmetrics['f1']))			# F1 Score
		mlflow.sklearn.log_model(model.model, "KerasModel")
