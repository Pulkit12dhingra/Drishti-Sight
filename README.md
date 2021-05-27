# Object-detection-with-feedback

## About the project
It is a project to elevate people with impaired eyesight. 
The project deals with object detection technology to detect obstacles in the way and give voice feedback. This project is developed to help people with impaired eyesight. It alerts the user about the upcoming obstacle, its distance from the user, and its location. This will help the user to move safely.
The project is deployed over a web API using flask as a backend.

![video_description](/gif1.gif)

# YOLO Object Detection
YOLO stands for You Only Look Once. It's an object detection model developed to overcome the drawbacks of R-CNNs and Fast R-CNNs models, mainly of low frame rates.

## Algorithm:
The algorithm that YOLO uses is called the One-stage detector strategy. These algorithms treat object detection as a regression problem, taking a given input image and simultaneously learning bounding box coordinates and corresponding class label probabilities.

## Working:
Before diving deep into the working of the YOLO algorithm, let's strengthen our understanding of object detection. Object detection or Object localization deals with the identification of objects in the frame and making a bounding box around the detected object. The difference between object detection and image classification is that in image classification, we get an output from our model whether the particular object is present in our frame or not. But in Object detection, we get the label associated with the object along with the height width and x and y location point of that object. This information enables us to draw a bounding box around that object. 

![single_object_YOLO_object_detection](/images/img1.jpg)

Additional information related to the position is provided in the training of a neural network to enable it to learn and generate an output of a vector having all the required information.

This process works well on frames having single objects, but it may cause an error while detecting multiple objects in a single frame since there may be n number of objects in the frame, so determining the size of the neural network is difficult.

The YOLO algorithm will divide the frame or image into a grid format, and for each grid cell, we can get an output vector associated with the object in that cell. This enables the algorithm to detect multiple objects in our frame.

![grid_object_YOLO_object_detection](/images/img2.jpg)

As we get vectors of grids for an object, we may get multiple bounding boxes with their respective probability. We cannot simply use the max probability one as there may be more than one object of a particular class in the frame. Thus, we use the concept of Intersection Over Union (IOU), which is to take the intersection area and union area of all the available boxes. 
IOU= intersection area / union area

<ul>
<li>If the IOU value is 1 then boxes are completely overlapping.</li>
<li>If the boxes are overlapping the value will be close to one.</li>
<li>If the boxes and not overlapping the value will be 0.</li>
</ul>

This will enable us to select the accurate bounding box by discarding the one with less IOU value.
This is also called NO max suppression.

# Data And Model 
The dataset which I use is from the COCO Dataset. The dataset contains  80 different classes, 330K images (>200K labeled),1.5 million object instances.
The Model used is pre-trained over the coco dataset. We have three separate files to load the weights, labels, and configuration files of the model.
<a href="https://drive.google.com/drive/folders/1XG59Uj9c_7EqiSQWMcTqtidynqj-G_yE?usp=sharing" target="_blank"> The weights can be downloaded from here</a> since it was a big file it can't be uploaded over github.
The model is built using OpenCV using the Deep neural Network module.

# Feedback Generation
After detecting the object, a response message is also generated based on the object's location in the frame. We can get an idea of the distance by calculating the area of the bounding box. If the box covers a small area then probably the object is distant. If the box covers a large area then probably the object is nearby.
We can also detect the location of the objects. In the area of width, the object may lie either on left, right, or center, and similarly for height it may be on top mid, or bottom. 

<ul>
<li>If we have the x coordinate our box center point. We can check its location to the center of the frame(horizontally) to know the location of the object over x coordinate</li>
<li>If we have the y-coordinate our box center point. We can check its location to the center of the frame(vertically) to know the location of the object over x coordinate</li>
</ul>

Finally, we can convert all this info to a sentence using string concatenation and use the GTTS module to generate an audio file.

To prevent the same audio to play multiple times, we store the current sentence and compare it with the previous sentence. If there is any change, then we'll play the new audio of the current sentence. This enables us to prevent the continuous playing of our audio file.

# Deployment
The project is deployed over the Web using Flask Rest API. We make use of HTML and CSS to built the front end and Flask API to get our backend ready for the project. 

<a href="https://github.com/Pulkit12dhingra/Object-Detection-with-Feedback">Click here to go on top</a>
