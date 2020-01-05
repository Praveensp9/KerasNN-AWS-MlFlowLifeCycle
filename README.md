# KerasNN-AWS-MlFlowLifeCycle
This is a Keras Neural Network Model which logs the metrics Precision, Recall, F1Score,Accuracy to MLflow( https://mlflow.org/ ) hosted on AWS Kubernetes cluster. It takes 3 Hyperparameters Epochs, batch size, Learning rate.


Python Version Used : Python 3.7.2

Kubernetes Cluster Link where mlflow server is running : <Your AWS Instance running Kubernetes>

command to run the code : python MlFlowModel.py --epoch {epoch} --batchsize {batch_size} --learningrate {learning_rate}
						  eg : python MlFlowModel.py --epoch 20 --batchsize 100 --learningrate 0.01
						  	  
Dataset Used for this Homework : IMDB dataset

******************************************************************************************************************************


The code written implements a Keras Neural Network Model with 3 Dense Layers and 2 Dropout layers and logs the Metrics and HyperParameters to the MlFlow Server on Kubernetes Cluster which is running on free AWS instance.

The 3 Hyperparameters (P1, P2, P3) used for this model are number of epochs, batch size and learning rate. The metrics that are calculated with the trained keras nn model on the validation data are Accuracy, Precision, Recall and F1 Score.


The Code consists of 3 Classes:

class LoadData :
		This class downloads the IMDB dataset and load the data. The data is normalized and then split into train set (80%) and test set (20%).
		Returns the train and test set.

class prepareKerasModel : 
		This class prepares the Keras model with the above mentioned dense and dropout layers , compile the model which is used for training the data.

class Metrics : 
		This class is used for calculating the precision, recall and f1 score of the results after the model is trained and tested on the validation data.
		Returns the Precision, Recall, F1 Score.

In the main method the 3 classes are called and the model is trained on the training data and validated on the test data.


MLFLOW logging: 

Finally, the Metrics calculated are Accuracy, Precision, Recall and F1 Score. These metrics and Hyperparameters are logged into the MlFlow Server running on kuberenetes cluster hosted on a free aws instance.

Total number of experiments conducted with varying epochs, batch size and learning rate  : 25



******************************************************************************************************************************


Keras model summary of layers:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 50)                1500050   
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                2550      
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                2550      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 51        
=================================================================
Total params: 1,505,201
Trainable params: 1,505,201
Non-trainable params: 0


******************************************************************************************************************************



