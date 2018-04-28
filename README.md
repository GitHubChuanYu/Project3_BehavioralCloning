# **Behavioral Cloning** 

## Writeup Report

### This is Chuan's writeup report for Udacity self-driving car nano degree program term 1 project 3 - behavioral cloning.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./exampleimages/Center.jpg "CenterLaneDriving"
[image2]: ./exampleimages/Recover1.jpg "RecoverDriving1"
[image3]: ./exampleimages/Recover2.jpg "RecoverDriving2"
[image4]: ./exampleimages/Recover3.jpg "RecoverDriving3"
[image5]: ./exampleimages/Recover4.jpg "RecoverDriving4"
[image6]: ./exampleimages/Flip1.jpg "Original Image"
[image7]: ./exampleimages/Flip2.jpg "Flipped Image"
[image8]: ./exampleimages/MultipleCameras.png "MultipleCameras"
[image9]: ./exampleimages/Corner1.jpg "Corner1"
[image10]: ./exampleimages/Corner2.jpg "Corner2"
[image11]: ./exampleimages/Corner3.jpg "Corner3"
[image12]: ./exampleimages/Corner4.jpg "Corner4"
[image13]: ./exampleimages/Corner5.jpg "Corner5"
[image14]: ./exampleimages/Corner6.jpg "Corner6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on powerful neural network architecture published by autonomous vehicle team in Nvidia. It contains five convolutional layers and 3 fully connected layers.  (model.py lines 76-86) 

The model includes RELU layers which are embedded in keras convolution layer to introduce nonlinearity (model.py lines 76-80), and the data is normalized in the model using a Keras lambda layer (model.py lines 72). 

#### 2. Attempts to reduce overfitting in the model

In my model, I did not use dropout layer to reduce overfitting due to the reason that dropout may cause the model to lose some important training data with large steering angle. And these large steering angle data do not have high probability in overall training dataset so dropout will easily cause the lost of them. But these pieces of traning dataset with large steering angle are critical for model to predict steering angle for curve driving and turning.

Instead I tried to get large amounts of training data to overcome the overfitting in the model. And it turns out to be very effective with very close training and validation loss. 

The total Train_Data size is about 276MB, so I cannot include it in this respository. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 89).

#### 4. Appropriate training data

The large amounts of training dataset contains these driving scenarios:
* Two laps of center lane driving with one clockwise and another counter-clockwise
* One lap of recovering driving from sides
* A lot of small data collection for smoothly driving through corners (even with both clockwise and counter-clockwise directions) A lot of fun!

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from simple network and then improve the network with more appropriate and complicate layers by trial and testing in simulator step by step.

My first step was to try a flatten layer connected with a single output node to see whether the model will drive the car autonomously. The single output node will predict the steering angle, which makes the network a regression type one.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had very high mean squared error on both training and validation dataset. This is mainly due to the simple regression network.

Then I tried a more powerful LeNet since I am familiar with it in previous class and it has convolutional layers. I adapt the input size and also output size since we only need one input instead of classification on 10 outputs. Both the mean squared errors on training and validation dataset reduced a lot. However after testing the model output in simulator, it turns out the car can drive autonmously straight ahead but not following the track very well. I think the main reason is that LeNet is designed for classification with several outputs instead of one single continuous output like the steering angle output in our case.

Lastly, as suggested in the class, I tried even more powerful network published by the autonomous team in Nvidia. This network contains more convolutional layers and fully connected layers and seems to be specifically designed for one single steering angle output. 

With this Nvidia network, I trained with my one lap of center-lane driving data and test the model output in simulator. It turns out I got both even lower mean squared errors in training and validation datasets. However, it seems mean squared error of training dataset is higher than that of validation dataset. This means overfitting. And the test result in simulator shows that vehicle can drive straight well following the track. However, when it encounters corner, it cannot turn quickly and enough to go through the corner, instead it keeps driving straight and eventually go offroad out of the track. 

As suggested in the class, if the model is overfitting, several tactics can be used to deal with that. I tried collecting more data and further augmenting the data instead of using dropout and using fewer convolution or fully connected layers. The reason is I don't want to loose critical and low probability data with large steering during dropout or using less convolution layers for driving through corners autonomously. So I tried collecting some additonal data including drving in opposite direction, smooth corner driving in both directions, and some recovery driving from off road back to track. Also I have applied data augentation method like flipping images and steering measurements, and using additional data from left and right cameras.

Eventually with more useful data collected, the trained model output can used to drive the vehicle autonomously in simulator and following the track pretty well!

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-86) is shown below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Layer1: Data Normalization     	| Keras lambda normalization, outputs 160x320x3 	|
| Layer2: Image Cropping	    | Crops off useless top and bottom parts of original image  |
| Layer3: Convolution 5x5	| 2x2 stride, output depth 24, activation "Relu" |
| Layer4: Convolution 5x5	| 2x2 stride, output depth 36, activation "Relu" |
| Layer5: Convolution 5x5	| 2x2 stride, output depth 48, activation "Relu" |
| Layer6: Convolution 5x5	| 1x1 stride, output depth 64, activation "Relu" |
| Layer7: Convolution 5x5	| 1x1 stride, output depth 64, activation "Relu" |
| Layer8: Flatten	| Flatten images for fully connected layers |
| Layer9: Fully connected		| outputs 100        									|
|	Layer10: Fully Connected		|				input 100, output 50								|
|	Layer11: Fully Connected		|				input 50, output 10								|
|	Layer12: Fully Connected		|				input 10, output 1								|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive through the corner smoothly. These images show what a recovery looks like:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Also, a large amount of data are collected for clockwise and counter-clockwise smooth curve driving so the trained model can tell vehicle how to drive through curves smoothly along the track. Example images for curve driving are shown here:

Left turn counter-clockwise curve driving:

![alt text][image9]
![alt text][image10]
![alt text][image11]

Right turn clockwise curve driving:

![alt text][image12]
![alt text][image13]
![alt text][image14]

To augment the data sat, I also flipped images and angles thinking that this would overcome the left turn bias of the data training track also add more different training data in different scenarios. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Also, I have utilized multiple camera images to mimic off-center driving data in real world. This will also help to train the vehicle to learn how to deal with off-center driving properply. This principle of using multiple camera images is demonstrated in this image:

![alt text][image8]

The codes for using multiple camera images are shown in model.py lines 37-45. The principle is for a left turn, a softer left turn is needed for left camera image, while a harder left turn is needed for right camera image. The steering angle correction parameters for left and right images can be set to tunable if needed.

After the collection process, I had 6947x6 = 41682 number of data points (6947 is the original number of data points, 6 times is due to 3 times of 3 cameras and 2 times of flipping data augmentation). I then preprocessed this data by using generators

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by checking the mean squared error for validation set starts to increase in epoch 4. I used an adam optimizer so that manually training the learning rate wasn't necessary. Finally the simulator test showed the vehicle can drive autonomously and smoothly through the track without any off-track problems, the result is shown in video.mp4.
