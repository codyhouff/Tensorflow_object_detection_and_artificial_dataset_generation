from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
 
import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


###########################################################

csv_input = 'data/train_labels.csv'
output_path = 'data/train.record'
image_dir = 'images/train'

csv_input_2 = 'data/test_labels.csv'
output_path_2 = 'data/test.record'
image_dir_2 = 'images/test'

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'dead_end_sign':
        return 1
    elif row_label == 'do_not_enter_sign':
        return 2
    elif row_label == 'interstate_sign':
        return 3
    elif row_label == 'no_u_turn_sign':
        return 4
    elif row_label == 'railroad_crossing_sign':
        return 5
    elif row_label == 'speed_limit_sign':
        return 6
    elif row_label == 'stop_sign':
        return 7
    elif row_label == 'street_name_sign':
        return 8
    elif row_label == 'yield_sign':
        return 9
    else:
        None

###########################################################

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.  read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


writer = tf.python_io.TFRecordWriter(output_path)
path = os.path.join(image_dir)
examples = pd.read_csv(csv_input)
grouped = split(examples, 'filename')
for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the train TFRecords: {}'.format(output_path))


writer = tf.python_io.TFRecordWriter(output_path_2)
path = os.path.join(image_dir_2)
examples = pd.read_csv(csv_input_2)
grouped = split(examples, 'filename')
for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path_2)
print('Successfully created the test TFRecords: {}'.format(output_path_2))


