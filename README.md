# Tensorflow object detection and artificial dataset generation
I reversed engineered the TensorFlow Object Detection API to make it easier to use, much simplier, and added voice. I created a artificial dataset generator which saves 100s of hours by avoiding having to find and label by hand, 1000s of images.

<p align="center">
  <img width="700" img src="results/images_for_readme/top_two_imgs.JPG">
</p>

# Motivation
The motivation for this project lays both personal interest in a better understanding for object detection and academic research. The goal is to develop a foundation for a road-sign-detection (RSD) with the option to add further objects or functions to it. The ultimate goal is to have a useable object detection for the automotive sector.

# Table of Contents

0. Setup
    * <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'>Tensorflow-gpu Installation</a><br>
    * download a model 
    * download a config file
    * Find front images
    * Prepare front images


1. Generate Training Images and csv file 
2. Create TF records
3. Train 
4. Tensorboard
5. Export inference graph
6. run model on images in jupyter notebook
7. run model on webcam 

# 0. Setup
### Tensorflow-gpu Installation
Install Tensorflow-gpu. Tutorial available <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'>here</a><br> or youtube tensorflow-gpu installation

### download a model 
download model from <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md'>here</a><br> I used the ssd_mobilenet_v1_coco model.

    .
    ├── models                                # models folder containing the models
        ├── ssd_mobilenet_v1_coco model       # put the model folder here 

### download a config file 
download the config file that matches the model from <a href='https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs'>here</a><br>
    
    .
    ├── training                                # training folder
        ├── ssd_mobilenet_v1_pets.config        # put the config file here
        ├── object-detection.pbtxt              

### Find front images
I chose street signs, but it can be anything.
<p align="center">
  <img width="500" img src="results/images_for_readme/signs_front_all.JPG">
</p>

### Prepare front images
You must remove the background. I used: https://www.remove.bg

<p align="center">
  <img width="500" img src="results/images_for_readme/remove_background1.jpg">
</p>

    .
    ├── front                                # models folder containing the models
        ├── street_signs                     # put the front street sign images in here
        ├── new folder                       # or if its another class of object, put the images in here

# 1. Generate Training Images and csv file 

The first step to take was to define the road signs and objects for the database. The database builds up on the RUB ["German Traffic Sign Database"][1], therefore the objects in the database used in the repository are similar pictures of everyday traffic situations in Germany. In order to build the database that would be able to detect a larger amount of road signs it was necessary to label a much larger number of pictures. The goal was to distinguish between more than 150 road signs, traffic lights and more than 15 physical objects such as pedestrians, cars and motorcycles.


### Analysing the Labels

As real-life traffic situations are used as input of the process the dataset needs to be reviewed and analysed regularly. Therefore, another tool was developed (DataSetCLI.py) to manage the large amounts of data. The tool offers multiple options for the database. All functions require a path to the root folder of your data, which should contain only images and xml-label files.

    .
    ├── root Data               #Root folder containing the data
        ├── images              #images location folder
        ├── labels              #xml location folder
        


### Generate Training Images

As not every class holds the same number of objects it becomes necessary to implement a data augmentation process. With this, existing pictures are alternated in such way that these can be used again in the learning process. For the augmentation the Python library "augmentor.py" [3] by the MIT is used. The tool has a large amount of functions implemented of which those useable for road sign detection are shown below. Some of these are only applayble to certain classes.

#### Rotate

Rotation is a simple function that allows to generate variance in the dataset. It needs to be ensured that the rotation is not to strong but somewhere between 10 and -10 degrees. Elsewise traffic signs may have a different meaning. This is also the reason why the rotate function of tensorflow can't be used as it only allows rotations at 90, 180 and 270 degrees.

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 35 27" src="results/images_for_readme/rotate_two_imgs.JPG">

#### Brightness

Rotation is a simple function that allows to generate variance in the dataset. It needs to be ensured that the rotation is not to strong but somewhere between 10 and -10 degrees. Elsewise traffic signs may have a different meaning. This is also the reason why the rotate function of tensorflow can't be used as it only allows rotations at 90, 180 and 270 degrees.

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 35 27" src="results/images_for_readme/brightness_two_imgs.JPG">

#### Relocate

The Zoom function is rather simple and lays focus on a different part of the picture. Yet the size of the image remains the same. The main advantage lays in a variance of quality and the relative strong change of objects in the overall image. 

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 48 06" src="results/images_for_readme/relocate_two_imgs.JPG">

#### Stretch

Stretch is useable for many different directions. It includes horizontal and vertical shearing as well as shearing to each of the corners. The function augments the data in such way as it would result if another picture was taken seconds later. It also makes the trained model more robust towards different angles.

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 40 54" src="results/images_for_readme/stretch_two_imgs.JPG">

#### Resize

Elastic distortion is a very interesting alteration of the pictures. As it can be seen on the right picture the object's corners, such as the large direction sign, are warped. This happens usually while driving when the car hits potholes or experiences other sudden and strong movements. Due to the image generation line by line the image gets distorted.

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 38 58" src="results/images_for_readme/resize_two_img.JPG">

#### Contrast

The function is very simple as it just alters the contrast of the image. The idea behind this is again improving the robustness of the trained model. Different contrasts occur usually in different lightning situations and the image quality of the used camera.

<img width="1489" alt="bildschirmfoto 2018-11-19 um 12 37 33" src="results/images_for_readme/contrast_two_imgs.JPG">

#### Color

Elastic distortion is a very interesting alteration of the pictures. As it can be seen on the right picture the object's corners, such as the large direction sign, are warped. This happens usually while driving when the car hits potholes or experiences other sudden and strong movements. Due to the image generation line by line the image gets distorted.

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 38 58" src="results/images_for_readme/color_two_imgs.JPG">

#### Results

Elastic distortion is a very interesting alteration of the pictures. As it can be seen on the right picture the object's corners, such as the large direction sign, are warped. This happens usually while driving when the car hits potholes or experiences other sudden and strong movements. Due to the image generation line by line the image gets distorted.

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 38 58" src="results/images_for_readme/results_multiple_images.JPG">

#### Generate csv file

With this a fairly large database was generated including 50.000 labels on approximately 35.000 images. As the objects, that were to be labelled, changed later on, the number of labels will keep growing rapidly. This will be done on the existing image database of 35.000 samples. An example of the database is presented below. 

<p align="center">
  <img width="700" img src="results/images_for_readme/csv_file_pic4.jpg">
</p>

# 2. Generate TF records

For this, two neural networks were taken into account. "Faster_R-CNN_Inception_V2_COCO" and "SSD_Mobilenet_COCO" both neural networks are pretrained on the COCO dataset that includes thousands of pictures with labels from everyday situations, such as humans, cars, trees, airplanes, etc. (http://cocodataset.org/#home)[6]. Yet both differ strongly.

<p align="center">
  <img width="700" img src="results/images_for_readme/csv_file_pic4.jpg">
</p>

# 3. Train

With this a fairly large database was generated including 50.000 labels on approximately 35.000 images. As the objects, that were to be labelled, changed later on, the number of labels will keep growing rapidly. This will be done on the existing image database of 35.000 samples. An example of the database is presented below. 

<p align="center">
  <img width="700" img src="results/images_for_readme/csv_file_pic4.jpg">
</p>

# 4. Tensorboard

With this a fairly large database was generated including 50.000 labels on approximately 35.000 images. As the objects, that were to be labelled, changed later on, the number of labels will keep growing rapidly. This will be done on the existing image database of 35.000 samples. An example of the database is presented below. 

<p align="center">
  <img width="700" img src="results/images_for_readme/graph_tensorboard2.JPG">
</p>

# 5. Export inference graph

With this a fairly large database was generated including 50.000 labels on approximately 35.000 images. As the objects, that were to be labelled, changed later on, the number of labels will keep growing rapidly. This will be done on the existing image database of 35.000 samples. An example of the database is presented below. 

<p align="center">
  <img width="700" img src="results/images_for_readme/csv.jpg">
</p>

# 5. Export inference graph



# 4. Results

The results need to be distinguished in multiple ways. Such as accuracy and the kind of failures that occur. Over the entire model we reached an accuracy of about 85% of the road signs, with the camera close to the object. Which is well above the initial expectations for this research topic. The failures that occur in object-detection can be separated into tour different modes. Those will be described quickly, as a starting point for further readings into the subject.

#### True-Positive

The object detector correctly identifies an object. Hence, this is the result we are seeking for and no failure occured. In the case of this object detection this result became more reliable the closer the road sign was. ![sign1](https://user-images.githubusercontent.com/34514578/51392469-1559bb00-1b70-11e9-8799-46f090e8c65e.jpg)


#### False-Positive

A false positive detection appears when a sign is marked incorrectly. Incorrectly marked signs usually occur if those are too small, so for example in a large distance. The other main reason is an insufficient amount of labels, saved in the database. ![falsepositive](https://user-images.githubusercontent.com/34514578/51392446-fe1acd80-1b6f-11e9-8c5b-a344e45467b3.jpg). This is especially important, as completely wrong detections can be hard to understand and to solve in many cases and sometimes even relate to issues within the label-database. Luckily this model doesn't suffer much from it. Usually high thresholds also ensure the limitation of true-negative detections. 

#### True-Negative

True negative detections are by definition not very spectacular, yet important. In this case the detector correctly doesn't give out any kind of result.

#### False-Negative

The opposite to true-negative are false-negative detections. In this case a sign or object is just missed by the detector. Also, often relating to low amounts of labels in the dataset. In this particular case we also see the limits of the faster-RCNN as shown in the picture below. While the "stop"-sign on the right and the "bus-stop"-sign are detected correctly, the "stop"-sign at the upper edge of the image is missed by the object detector. In this case the stop-sign is not at a common place but at a very usual space. The RCNN misses that due to its focus on common sizes and positions for the initial choice of regions to be analysed afterwards.

![nostop](https://user-images.githubusercontent.com/34514578/51392458-0a068f80-1b70-11e9-8178-c0395abd82e1.jpg)

# 5. Outlook

The object-detection still needs further improvements in many cases. It is yet not accurate enough nor does the speed match our demands. As this was archived within a term-paper, it is still a strong start for further improvements. Those will include the database as well as tests with other neural networks such as YOLO. Furthermore, reinforced learning needs to be taken into account.

For further questions please refer to our LinkedIn profiles (that you can find in our profiles), contact us here on GitHub. We also appreciate if you leave a comment. 

# List of Refrences
[1]http://benchmark.ini.rub.de/?section=gtsdb&subsection=news
[2]https://github.com/tzutalin/labelImg
[3]https://augmentor.readthedocs.io
[4]https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/README.md
[5]https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[6] http://cocodataset.org/#home
