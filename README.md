# Tensorflow object detection and artificial dataset generation
I reversed engineered the TensorFlow Object Detection API to make it easier to use, much simplier, and added voice. I also created a artificial dataset generator which saves 100s of hours by avoiding having to find and label by hand, 1000s of images. Instead it can generate and label 1000s of artificial images in seconds.

<p align="center">
  <img width="700" img src="results/images_for_readme/top_two_imgs.JPG">
</p>

# Motivation
The current Tensorflow object detection from github is increadibly unorganized and time consuming to use. You have to go through 20+ steps each time you want to train and use a model. The most time consuming part is probably finding the training images and labeling each one of them by hand. It becomes impractical once the number of classes reach more than five. Say for each class you have 800 training images, thats 4000 images you have to label by hand!

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

#### Edit config file and object-detection.pbtxt file (if necessary)

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


#### Edit _1_generate_training_images.py (if necessary)
```
background_directory = "backgrounds/(799)_medium/"  # path to backgrounds, (799) means 799 images per class of object 

images_directory = "images/"                        # output path to the test and train image folder

front_directory = "front/street_signs/"             # path to the front object images 
```
#### Edit _2_generate_tfrecord.py (if necessary)
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
#### Edit _3_train.py (if necessary)
```
train_dir = "training/training_results"                         # path for output folder with the train data

pipeline_config_path = "training/ssd_mobilenet_v1_pets.config"  # path to the config file
```
#### Edit _4_tensorboard.py (if necessary)
```
tensorboard_link = "http://LAPTOP-M6D3SOR6:6006/" # link to tensorboard, your personal link will be on the cmd prompt 

training_directory = "training/"                  # path to the folders where train iteration info goes
```
#### Edit _5_export_inference_graph.py (if necessary)
```
pipeline_config_path = "training/ssd_mobilenet_v1_pets.config"                      # path to config file

trained_checkpoint_prefix = "training/training_results_3(old)/model.ckpt-45464"     # path to a model checkpoint of your choosing, choose one with a low loss

output_directory = "inference_graphs/stop_sign_generated_images_inference_graph_4"  # path to output inference graph
```
#### Edit _6_run_model_jupyter_notebook.ipynb (if necessary)
```
MODEL_NAME = 'inference_graphs/stop_sign_generated_images_inference_graph_3(old)'    # path to inference graph

PATH_TO_TEST_IMAGES_DIR = 'test_images/street_signs'				     # path to test images

number_test_images = 18							             # number of images in the test_images folder
```
#### Edit _7_run_model_webcam.py (if necessary)
```
MODEL_NAME = 'inference_graphs/stop_sign_generated_images_inference_graph_3(old)'    # path to inference graph
```



# 1. Generate Training Images and csv file 
Generates artificial training images and a csv file that contains the bounding box location and the class of the object. Also Variance is added to each image, this is to make the model more robust and able to handle images with poor lighting or bad angles.

#### Rotate

Rotates the front image a random amount between -15 to 15 degrees.

<img width="1555" src="results/images_for_readme/rotate_two_imgs.JPG">

#### Brightness

Darkens or lightens the front image a random amount between 10% to 130%. Also darkens or lightens the background a random amount.

<img width="1555" src="results/images_for_readme/brightness_two_imgs.JPG">

#### Relocate

Relocates the front image randomly on the background image. 

<img width="1555" src="results/images_for_readme/relocate_two_imgs.JPG">

#### Stretch

20% chance for a random amount of verticle stretch, 20% chance for a random amount of horizontal stretch, and 60% chance for no change.

<img width="1555" src="results/images_for_readme/stretch_two_imgs.JPG">

#### Resize

Resize the front image randomly between 30% to 6%.

<img width="1555" src="results/images_for_readme/resize_two_img.JPG">

#### Contrast

Changes the contrast of the front and backgrond image randomly between 10% to 200% 

<img width="1489" src="results/images_for_readme/contrast_two_imgs.JPG">

#### Color

Changes the contrast of the front and backgrond image randomly between 10% to 200%

<img width="1555" alt="bildschirmfoto 2018-11-19 um 12 38 58" src="results/images_for_readme/color_two_imgs.JPG">

### Results
Sample of the images that are generated.
<p align="center">
  <img width="500" img src="results/images_for_readme/results_gif1.gif">
</p>

### Generate csv file

Creates a train and test csv file that contains the bounding box location and the class of the object. Located in the data
file.
<p align="center">
  <img width="700" img src="results/images_for_readme/csv_file_pic4.jpg">
</p>

# 2. Generate TF records

Takes the train csv, test csv, train images, test images, and creates a test.record and a train.record. If it works correctly it will display the following:

```
Successfully created the train TFRecords: data/train.record
Successfully created the test TFRecords: data/test.record
```

# 3. Train
Takes the train.record, test.record, train images, the config file, the pbtxt file, the model and saves checkpoints in a folder in training. One of these checkpoints will be chosen to create the inference graph. If it works correctly it will display the following:

```
I0918 03:27:00.565701 26700 learning.py:507] global step 1: loss = 47.7762 (11.987 sec/step)
I0918 03:27:01.375576 26700 learning.py:507] global step 2: loss = 41.2166 (0.634 sec/step)
I0918 03:27:02.020375 26700 learning.py:507] global step 3: loss = 36.5167 (0.643 sec/step)
I0918 03:27:02.670631 26700 learning.py:507] global step 4: loss = 34.1669 (0.646 sec/step)
I0918 03:27:03.316424 26700 learning.py:507] global step 5: loss = 31.5301 (0.643 sec/step)
I0918 03:27:03.965198 26700 learning.py:507] global step 6: loss = 29.6940 (0.647 sec/step)
I0918 03:27:04.600069 26700 learning.py:507] global step 7: loss = 27.4882 (0.633 sec/step)
```

# 4. Tensorboard
Click on the _4_tensorboard.py file to open tensorboard and visualize the data.

<p align="center">
  <img width="700" img src="results/images_for_readme/graph_tensorboard2.JPG">
</p>

# 5. Export inference graph
Takes the config file, and checkpoint ex: "model.ckpt-45464" and creates a inference graph in the inference graphs folder. You must chose a checkpoint based on tensorboard and iteration. Try to wait till the total loss in the tensorboard graph is below 1.
```
trained_checkpoint_prefix = "training/training_results_3(old)/model.ckpt-45464"     # path to a model checkpoint of your choosing, choose one with a low loss
```

# 6. Run Model on Images in Jupyter Notebook
Make sure the MODEL_NAME in the _6_run_model_jupyter_notebook.ipynb is the path to the correct inference graph. 

```
MODEL_NAME = 'inference_graphs/stop_sign_generated_images_inference_graph_3(old)'    # path to inference graph
```
# 7. Run Model on Webcam
Make sure the MODEL_NAME in the _7_run_model_webcam.py is the path to the correct inference graph. 

```
MODEL_NAME = 'inference_graphs/stop_sign_generated_images_inference_graph_3(old)'    # path to inference graph
```

# Results


# Outlook

For further questions please refer to my LinkedIn profile or contact us here on GitHub. 

# List of Refrences
[1]https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
