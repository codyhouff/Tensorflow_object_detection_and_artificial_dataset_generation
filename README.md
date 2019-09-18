# Tensorflow object detection and artificial dataset generation
I reversed engineered the TensorFlow Object Detection API to make it easier to use, much simplier, and added voice. I created a artificial dataset generator which saves 100s of hours by avoiding having to find and label by hand, 1000s of images.

<p align="center">
  <img width="700" img src="results/images_for_readme/top_two_imgs.JPG">
</p>

# Motivation
The motivation for this project lays both personal interest in a better understanding for object detection and academic research. The goal is to develop a foundation for a road-sign-detection (RSD) with the option to add further objects or functions to it. The ultimate goal is to have a useable object detection for the automotive sector.

# Table of Contents

0. Setup
    * <a href='https://www.tensorflow.org/install/gpu'>Tensorflow-gpu Installation</a><br>
    * Find Front Images
    * Prepare Front Images
    * Get Test Images
    * <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md'>Download a Model</a><br>
    * <a href='https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs'>Download a Config File</a><br>
    * Edit Files Below if Necessary
    	* Edit Config File and Object-detection.pbtxt File
	* Edit _1_generate_training_images.py
	* Edit _2_generate_tfrecord.py
	* Edit _3_train.py
	* Edit _4_tensorboard.py
	* Edit _5_export_inference_graph.py
	* Edit _6_run_model_jupyter_notebook.ipynb
	* Edit _7_run_model_webcam.py

1. Generate Training Images and csv file 
2. Create TF records
3. Train 
4. Tensorboard
5. Export inference graph
6. Run Model on Images in Jupyter Notebook
7. Run Model on Webcam 

# Setup
### Tensorflow-gpu Installation
Install Tensorflow-gpu. Tutorial available <a href='https://www.tensorflow.org/install/gpu'>here</a><br> 
or youtube tensorflow-gpu installation.
Download Tensorflow dependencies <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'>here</a><br> and other dependencies: 
```
pip install Cython
pip install contextlib2
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib==3.0.1   # Had trouble with the newer versions for some reason

pip install PIL
pip install random
pip install glob
pip install os
pip install csv
pip install numpy
pip install pyttsx3
pip install cv2
```

### Find front images
I chose street signs, but it can be anything.
<p align="center">
  <img width="500" img src="results/images_for_readme/signs_front_all.JPG">
</p>

### Prepare front images
You must remove the background. I used: https://www.remove.bg.
Then name the picture the name of the object class

<p align="center">
  <img width="500" img src="results/images_for_readme/remove_background1.jpg">
</p>

    .
    ├── front                                # models folder containing the models
        ├── street_signs                     # put the front street sign images in here
        ├── new folder                       # or if its another class of object, put the images in here

### Get some test images and put them in a folder in test images 
save images as: image1.jpg, image2.jpg, etc

    .
    ├── test_images             
        ├── street_signs	# folder containing street signs test images
        ├── new folder          # new folder containing test images 

### Download a model 
download model from <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md'>here</a><br> I used the ssd_mobilenet_v1_coco model.

    .
    ├── models                                # models folder containing the models
        ├── ssd_mobilenet_v1_coco model       # put the model folder here 

### Download a config file 
download the config file that matches the model from <a href='https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs'>here</a><br>
    
    .
    ├── training                                # training folder
        ├── ssd_mobilenet_v1_pets.config        # put the config file here and edit
        ├── object-detection.pbtxt              # edit 

### Edit files below if necessary

#### Edit config file and object-detection.pbtxt file

Edit these lines in the config file in the train folder 
```
num_classes: 9					    # number of different objects the model will detect
	
fine_tune_checkpoint: "models/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt" # path to the model

input_path: "data/train.record"			    # path to the train.record file

label_map_path: "training/object-detection.pbtxt    # path to the pbtxt file

num_examples: 972				    # num of training images in the images/train folder after generating the images

input_path: "data/test.record"                      # path to the test.record file

label_map_path: "training/object-detection.pbtxt"   # path to the pbtxt file
```

Then edit the object-detection.pbtxt file in the training folder to list all the objects:
```
item {
  id: 1
  name: 'dead_end_sign'
}

item {
  id: 2
  name: 'do_not_enter_sign'
}
etc...
```


#### Edit _1_generate_training_images.py
```
background_directory = "backgrounds/(799)_medium/"  # path to backgrounds, (799) means 799 images per class of object 

images_directory = "images/"                        # output path to the test and train image folder

front_directory = "front/street_signs/"             # path to the front object images 
```
#### Edit _2_generate_tfrecord.py
```
path_to_csv = "data/"                # path to train test and train csv files
path_to_images = "images/"           # path to test and train image folders
output_path_record = "data/"         # output path for test and train record files

# TO-DO replace this with label map  # change names of object classes to the ones you are using
def class_text_to_int(row_label):    # make sure to use the same numbers to classes as in the object-detection.pbtxt file 
    if row_label == 'dead_end_sign':
        return 1
    elif row_label == 'do_not_enter_sign':
        return 2
    elif row_label == 'interstate_sign':
        return 3
    etc........
    else:
        None
```
#### Edit _3_train.py
```
train_dir = "training/training_results"                         # path for output folder with the train data

pipeline_config_path = "training/ssd_mobilenet_v1_pets.config"  # path to the config file
```
#### Edit _4_tensorboard.py
```
tensorboard_link = "http://LAPTOP-M6D3SOR6:6006/" # link to tensorboard, your personal link will be on the cmd prompt 

training_directory = "training/"                  # path to the folders where train iteration info goes
```
#### Edit _5_export_inference_graph.py
```
pipeline_config_path = "training/ssd_mobilenet_v1_pets.config"                      # path to config file

trained_checkpoint_prefix = "training/training_results_3(old)/model.ckpt-45464"     # path to a model checkpoint of your choosing, choose one with a low loss

output_directory = "inference_graphs/stop_sign_generated_images_inference_graph_4"  # path to output inference graph
```
#### Edit _6_run_model_jupyter_notebook.ipynb
```
MODEL_NAME = 'inference_graphs/stop_sign_generated_images_inference_graph_3(old)'    # path to inference graph

PATH_TO_TEST_IMAGES_DIR = 'test_images/street_signs'				     # path to test images

number_test_images = 18							             # number of images in the test_images folder
```
#### Edit _7_run_model_webcam.py
```
MODEL_NAME = 'inference_graphs/stop_sign_generated_images_inference_graph_3(old)'    # path to inference graph
```

# 1. Generate Training Images and csv file 

The first step to take was to define the road signs and objects for the database. The database builds up on the RUB ["German Traffic Sign Database"][1], therefore the objects in the database used in the repository are similar pictures of everyday traffic situations in Germany. In order to build the database that would be able to detect a larger amount of road signs it was necessary to label a much larger number of pictures. The goal was to distinguish between more than 150 road signs, traffic lights and more than 15 physical objects such as pedestrians, cars and motorcycles.


### Add variance to the images

As not every class holds the same number of objects it becomes necessary to implement a data augmentation process. With this, existing pictures are alternated in such way that these can be used again in the learning process. For the augmentation the Python library "augmentor.py" [3] by the MIT is used. The tool has a large amount of functions implemented of which those useable for road sign detection are shown below. Some of these are only applayble to certain classes.

#### Rotate

Rotates the front image a random amount between -15 to 15 degrees.

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 35 27" src="results/images_for_readme/rotate_two_imgs.JPG">

#### Brightness

Darkens or lightens the front image a random amount between 10% to 130%. Also darkens or lightens the background a random amount.

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

### Results

Elastic distortion is a very interesting alteration of the pictures. As it can be seen on the right picture the object's corners, such as the large direction sign, are warped. This happens usually while driving when the car hits potholes or experiences other sudden and strong movements. Due to the image generation line by line the image gets distorted.

<p align="center">
  <img width="500" img src="results/images_for_readme/results_gif1.gif">
</p>

### Generate csv file

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

# 6. Run Model on Images in Jupyter Notebook

# 7. Run Model on Webcam


# Results


# Outlook

The object-detection still needs further improvements in many cases. It is yet not accurate enough nor does the speed match our demands. As this was archived within a term-paper, it is still a strong start for further improvements. Those will include the database as well as tests with other neural networks such as YOLO. Furthermore, reinforced learning needs to be taken into account.

For further questions please refer to our LinkedIn profiles (that you can find in our profiles), contact us here on GitHub. We also appreciate if you leave a comment. 

# List of Refrences
[1]http://benchmark.ini.rub.de/?section=gtsdb&subsection=news
[2]https://github.com/tzutalin/labelImg
[3]https://augmentor.readthedocs.io
[4]https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/README.md
[5]https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[6] http://cocodataset.org/#home
