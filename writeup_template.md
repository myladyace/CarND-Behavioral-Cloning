#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 demonstrating an autonomous driving example with the model.h5

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model simply uses the nvidia network provided in the lecture containing 5 convolutional layers and 4 fully connected layers.(model.py lines 74-83) 

The model includes RELU layers to introduce nonlinearity (code line 74-78)

The data is normalized and croped in the model using a Keras lambda and crop layer (code line 72-73). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 13-14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Actually I found the sample training data is good enough to make the car stay on the road so I did not add additional data.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was following the steps given in the lecture.

My first step was to preprocess the data and use the LeNet, I thought this model might be appropriate and the loss on both training data and validation is low. But when I run the simulator, the car still cannot handle big curve. 

Here I should have two options, one is to add more curve data and another is to try some other network, I chose both by adding some data based on current data and implementing an even powerful network.

I then used flip and multiple cameras to augment the data size and also implemented the nVidia Autonomous Car Group model. However, while running the code, it came out memory error, so I used the generator to save memory. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 71-83) is shown below with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| outputs 160x320x3  							| 
| Crop        		| outputs 90x320x3 							| 
| Normalization         		| outputs 90x320x3  							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 43x158x24 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 20x77x36   |
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 8x37x48   |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 6x35x64   |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x33x64   |
| RELU					|												|
| Flatten		      	| outputs 8448									|
| Fully connected		| outputs 100       							|
| Fully connected		| outputs 50  									|
| Fully connected		| outputs 10        							|
| Fully connected		| outputs 1        							|



####3. Creation of the Training Set & Training Process

I simply uses the sample data but using flipping and all three cameras to augment the data size to about 38500 images.

Then I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 because when I set it to 5, I found the validation loss of the forth epoch is greater than the third. I used an adam optimizer so that manually training the learning rate wasn't necessary.
