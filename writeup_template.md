# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn_architecture.png "Model Visualization"
[image2]: ./examples/round1_center.jpg "Recovery Image"
[image3]: ./examples/round1_right.jpg "Recovery Image"
[image4]: ./examples/round3.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 18-24). These layers are followed by batch normalization laayers and relu activation functions.

The model includes 4 Dense layers after the 5 Conv Layers, with 100, 500, 10 and 1 neurons (model.py lines 18-24).


#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting (model.py lines 158 & 163). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 100-105). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, which combines the advantages of AdaGrad (maintains a per-parameter learning rate that improves performance on problems with sparse gradients) and RMSProp (maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weight), so the learning rate was not tuned manually (model.py line 129).



#### 4. Appropriate training data

The simulation was run 4 times on the first map collect data. There are 2 normal runs, 1 backward run and 1 with a lot of off the road drivings with corrections.

The training data was then get flipped horizontally for augmentation.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by adding and subtracing 0.2 from the measurment from the center image.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
I implemented the model architecture based on: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf


My first step was to use a convolution neural network model with 2 convolutional layers with dropout. I thought this model might be appropriate because two layers of convolutions can capture some relatively complex shapes, e.g. circles and squares. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture  consisted of a convolution neural network with the following layers and layer sizes :


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Convolution 5x5     	| 24x5x5 	                                    |
| BatchNormalization	|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride			                        |
| Convolution 5x5     	| 36x5x5 	                                    |
| BatchNormalization	|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride			                        |
| Convolution 5x5     	| 48x5x5 	                                    |
| BatchNormalization	|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride			                        |
| Convolution 5x5     	| 64x3x3 	                                    |
| BatchNormalization	|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride			                        |
| Convolution 5x5     	| 64x3x3 	                                    |
| BatchNormalization	|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride			                        |
| Flatten				|												|
| Fully Connected		| Output size 100				                |
| RELU					|												|
| Dropout			    | Probability = 0.5							    |
| Fully Connected		| Output size 50			                	|
| RELU					|												|
| Fully Connected		| Output size 10			                	|
| RELU					|												|
| Dropout			    | Probability = 0.7								|
| Fully Connected		| Output size 1				                    |
 



![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

![alt text][image2]

To make the data symmetric, 1 backward run and 1 with a lot of off the road drivings with corrections.

![alt text][image4]

The last run was different by introducing errors and corrections, because in real life, the model needs to know what to do when the car gets close to the edge of the road or curbs. 

I random shuffled the data set and put 20% of the original data into a validation set. 

Next,  multi-camera images were used in the training set only . In order to recover from the left and right sides of the road, adding and subtracing 0.2 from the measurment from the center image was calculated.

![alt text][image3]


Finally, all the images from the center, left and right cameras are flipped horizontally, and the target is multiplied by -1. This applied to both traning and validation data sets.

After the collection process, I had 53262 number of images for training and 4438 images for validation (validation set is flipped but only the center images were used).

I then preprocessed this data. I firstly noramlized image = image / 255 - 0.5. Then chopped the irrelevant pixels.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
