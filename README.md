# Tensorflow object detection and artificial dataset generation
I reversed engineered the TensorFlow Object Detection API to make it easier to use, much simplier, and added voice. I also created a artificial dataset generator which saves 100s of hours by avoiding having to find and label by hand, 1000s of images. Instead it can generate and label 1000s of artificial images in seconds.

<p align="center">
  <img width="700" img src="results/images_for_readme/top_two_imgs.JPG">
</p>

# Motivation
The current Tensorflow object detection from github is increadibly unorganized and time consuming to use. You have to go through 20+ steps each time you want to train and use a model. The most time consuming part is probably finding the training images and labeling each one of them by hand. It becomes impractical once the number of classes reach more than five. Say for each class you have 800 training images, thats 4000 images you have to label by hand!

# Table of Contents

0. <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md'>Setup</a><br>
1. Generate Training Images and csv file 
2. Create TF records
3. Train 
4. Tensorboard
5. Export inference graph
6. Run Model on Images in Jupyter Notebook
7. Run Model on Webcam 

# 0. Setup
### <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md'>Click Here for Setup</a><br>

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
Takes the train.record, test.record, train images, the config file, the pbtxt file, the model and saves checkpoints in a folder in training. One of these checkpoints will be chosen to create the inference graph. 

Don't forget to change the num_examples to the number of images in the images/train folder.
```
num_examples: 972	# num of training images in the images/train folder after generating the images
```

If it works correctly it will eventually display something like this:
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
Click on the _4_tensorboard.py file to open tensorboard and visualize the data. Look at the Total loss.

<p align="center">
  <img width="700" img src="results/images_for_readme/graph_tensorboard2.JPG">
</p>

# 5. Export inference graph
Takes the config file, and checkpoint ex: "model.ckpt-45464" and creates a inference graph in the inference graphs folder. You must chose a checkpoint based on tensorboard and iteration. Try to wait till the total loss in the tensorboard graph is below 1.
```
trained_checkpoint_prefix = "training/training_results_3(old)/model.ckpt-45464"           # path to a model checkpoint of your choosing, choose one with a low loss
output_directory = "inference_graphs/stop_sign_generated_images_inference_graph_3(old)"   # path to output inference graph and choose the name of the graph
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

<p align="center">
  <img width="700" img src="results/images_for_readme/stop_sign_live.gif">
</p>

<p align="center">
  <img width="700" img src="results/images_for_readme/do_not_enter_sign.gif">
</p>


