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

output_directory = "inference_graphs/stop_sign_generated_images_inference_graph_4"  # path to output inference graph and choose the name of the graph
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

