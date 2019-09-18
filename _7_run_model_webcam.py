import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

cap = cv2.VideoCapture(0)

import pyttsx3

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util

from utils import visualization_utils as vis_util



######################################################################

# What model to download.
MODEL_NAME = 'inference_graphs/stop_sign_generated_images_inference_graph_3(old)'

######################################################################



# What model to download.
#MODEL_NAME = 'stop_sign_generated_images_inference_graph_3'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# In[9]:

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# In[ ]:


def voice(classes):
  signs_detected = " "
  signs_count = 0
  for i in range(10): 

      if ((int(classes[0][i]) == 1) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "dead end sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)            
            if (i >=1):
                signs_detected = signs_detected + " and "               
            signs_detected = signs_detected + " dead end sign "
            signs_count = signs_count + 1

      if ((int(classes[0][i]) == 2) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "do not enter sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)           
            if (i >=1):
                signs_detected = signs_detected + " and "               
            signs_detected = signs_detected + " do not enter sign "
            signs_count = signs_count + 1
            
      if ((int(classes[0][i]) == 3) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "interstate sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)           
            if (i >=1):
                signs_detected = signs_detected + " and "               
            signs_detected = signs_detected + " interstate sign "
            signs_count = signs_count + 1

            
      if ((int(classes[0][i]) == 4) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "no u turn sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)            
            if (i >=1):
                signs_detected = signs_detected + " and "               
            signs_detected = signs_detected + " no u turn sign "
            signs_count = signs_count + 1

      if ((int(classes[0][i]) == 5) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "railroad crossing sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)
            if (i >=1):
                signs_detected = signs_detected + " and " 
            signs_detected = signs_detected + " railroad crossing sign "
            signs_count = signs_count + 1

      if ((int(classes[0][i]) == 6) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "speed limit sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)           
            if (i >=1):
                signs_detected = signs_detected + " and "               
            signs_detected = signs_detected + " speed limit sign "
            signs_count = signs_count + 1
            
      if ((int(classes[0][i]) == 7) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "stop sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)
            if (i >=1):
                signs_detected = signs_detected + " and "               
            signs_detected = signs_detected + " stop sign "
            signs_count = signs_count + 1
            
            
      if ((int(classes[0][i]) == 8) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "street name sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print)            
            if (i >=1):
                signs_detected = signs_detected + " and "    
            signs_detected = signs_detected + " street name sign "
            signs_count = signs_count + 1

            
      if ((int(classes[0][i]) == 9) & (scores[0][i] >= 0.40)):
            percent_accuracy = int(scores[0][i] * 100)
            signs_print = "yield sign detected at " + str(percent_accuracy) + " percent accuracy"
            print(signs_print) 
            if (i >=1):
                signs_detected = signs_detected + " and "    
            signs_detected = signs_detected + " yield sign "
            signs_count = signs_count + 1

  if (signs_count > 0):
      signs_detected = signs_detected + " detected "        
      engine = pyttsx3.init()
      engine.say(signs_detected)
      engine.runAndWait()
      print("______________________________________________________________")    

while True:

  ret, image_np =cap.read()
  
  image_np_expanded = np.expand_dims(image_np, axis=0)
    
  with detection_graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_np_expanded.shape[1], image_np_expanded.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      (boxes, scores, classes, num) = sess.run(
         [detection_boxes, detection_scores, detection_classes, num_detections],
         feed_dict={image_tensor: image_np_expanded})   


  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),


      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8,
      min_score_thresh=0.40)

  voice(classes)

  cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
  if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
