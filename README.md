# KerasNN-AWS-MlFlowLifeCycle
This is a Keras Neural Network Model which logs the metrics Precision, Recall, F1Score,Accuracy to MLflow( https://mlflow.org/ ) hosted on AWS Kubernetes cluster. It takes 3 Hyperparameters Epochs, batch size, Learning rate.


Python Version Used: Python 3.7.2

command to run the code: 

python MlFlowModel.py --epoch {epoch} --batchsize {batch_size} --learningrate {learning_rate}

	e.g.: python MlFlowModel.py --epoch 10 --batchsize 500 --learningrate 0.01
	Note: Install the dependencies before executing the code.

Dataset used for this Homework: Keras IMDB dataset

******************************************************************************

* Code Implementation:

The code implements a Keras neural network model with an input layer,2 hidden layers,2 dropout layers and a output layer. It logs all the Metrics (4) and Hyperparameters (3) to the MlFlow Server running on Kubernetes Cluster which is hosted on free AWS instance.

The 3 Hyperparameters (P1, P2, P3) used for this model are number of epochs, batch size and the learning rate. The metrics that are calculated with the trained keras neural network model on the validation data are Accuracy, Precision, Recall and F1 Score.

The implementation of the code is as follows:

The Code consists of three classes which are called in the main method. The three classes are explained briefly as below.

class LoadData :

		This class downloads the IMDB dataset and prepares the data for training and testing. The data is normalized and then split into train set (80%) and test set (20%). It returns the train and test set.

class prepareKerasModel : 

		This class prepares the Keras neural network model with the input, hidden and output layers. I have used dense at every layer to make sure the units are fully connected and used 2 dropout layers to prevent overfitting. Later compile the keras model which is used for training the data. The input and hidden layers use 'relu' as the activation function. Since it is binary classification where the output is positive (1) or negative (0), I have used 'sigmoid' as the activation function for the output layer.
class Metrics: 

This class is used for calculating the precision, recall and f1 score of the results after the model is trained and tested on the validation data. It returns the Precision, Recall and F1 Score metrics. I have used sklearn metrics to calculate the f1 score, precision and recall in this method. This class is object is called in the callback’s parameter in the keras model fit method.

In the Main method, the 3 classes are instantiated and called. The model is trained on the training data and validated using the test data. I have used mlflow log_param, log_metric to log the metrics and hyper parameters to my MLFlow Server running on Kubernetes cluster in the main method.

Total parameters: 1,505,201
Trainable parameters: 1,505,201

* MLFLOW logging:

The metrics calculated are Accuracy, Precision, Recall and F1 Score. These metrics and Hyperparameters are logged into the MlFlow Server running on Kubernetes cluster hosted on a free aws instance.

Total number of experiments conducted with varying epochs, batch size and learning rate: 23

Based on these different experiments, the best set of hyper parameters combination which yields a very good F1 Score and Accuracy are given table below:
									
Hyper Parameters	Metrics
epochs         : 10	    F1 Score     : 0.894
batch size      : 500	  Accuracy    : 0.88
learning rate  : 0.01	    Recall          : 0.901
					 

It is very helpful that using this MLFlow platform, we can determine a set of hyper parameters that result in a good F1-score based on conducting different experiments as shown in the above table. 

Hence, for logging all our model metrics and their corresponding hyper parameters it is very good to use this MLFlow platform in building an end to end ML model. Based on different experiment metrics results, we can decide what are the best hyper parameters that results in a very good best f1 score and accuracy.


******************************************************************************
* Creating a Kubernetes Cluster and Running MlFLow Server:

Steps:

	1. Create a Kubernetes Cluster on free AWS EC2 instance.

	2. Install Mlflow (1.5.0) using (pip install mlflow) and it's dependencies.
	
	3. Create a S3 bucket to store your Mlflow logs, metrics, model and artifacts.
	
	4. Move your Running Mlflow Server to the Kubernetes Cluster by executing the   
                  following command:
		
                 e.g.:  mlflow server --default-artifact-root s3://<bucket_name>/ --host 0.0.0.0

	  The above command starts running mlflow server on the Kubernetes cluster on port            
               5000 and the artifacts, metrics data is stored to the aws s3 bucket mentioned in the        
               command.

	5. Set up a nginx reverse proxy server listening on port 5000 so that any calls to the 
                  Kubernetes cluster are redirected to mlflow server running on port 5000.

	6. Now use the Kubernetes cluster hostname (ec2-3-17-205-96.us-east-
                 2.compute.amazonaws.com) as your mlflow tracking uri in the code to log all your 
                mlflow metrics and hyperparameters.

	   e.g.:  mlflow.set_tracking_uri(ec2-3-17-205-96.us-east-2.compute.amazonaws.com)

	7. Once the setup is done, you can go to the Kubernetes cluster running on aws instance  
                  to check your mlflow metrics and start using the MLFlow platform.

******************************************************************************

