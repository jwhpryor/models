from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import csv
import tensorflow as tf
import numpy as np

np.random.seed(0xabcdef)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('label_dictionary', 'data/imgs/driver_imgs_list.csv',
                           """Dictionary of all filenames and their respective class info""")
'''
tf.app.flags.DEFINE_string('train_dir', 'data/imgs/train',
                           """Images train dir""")
tf.app.flags.DEFINE_string('eval_imgs_dir', 'data/imgs/test',
                           """Images train dir""")
                           '''
tf.app.flags.DEFINE_float('holdout', 0.1,
                          """Percent of samples to holdout.""")

# Make all of our sets nicely fit to the batch size
# (possibly losing a few samples)
BATCH_SIZE = 128
HOLDOUT = 0.1
TOTAL_SAMPLES = 22424
NUM_TRAIN_SAMPLES = int(TOTAL_SAMPLES * (1-HOLDOUT))
NUM_TRAIN_SAMPLES -= NUM_TRAIN_SAMPLES % BATCH_SIZE
NUM_EVAL_SAMPLES = (TOTAL_SAMPLES - NUM_TRAIN_SAMPLES)
NUM_EVAL_SAMPLES -= NUM_EVAL_SAMPLES % BATCH_SIZE

class DriverRecord:
    def __init__(self, filename, driver_id, class_id):
        self.filename = filename
        self.driver_id = driver_id
        self.class_id = class_id

# parses the csv provided by kg for training
def get_label_dic():
    output_dic = {}
    if not tf.gfile.Exists(FLAGS.label_dictionary):
        raise Exception('No label dictionary found')
    with open(FLAGS.label_dictionary, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            driver_id = int(row[0][1:])
            class_id = int(row[1][1:])
            filename = row[2]
            output_dic[filename] = DriverRecord(filename, driver_id, class_id)

    return output_dic

def get_train_eval_and_label(train=True):
    train_filenames, eval_filenames = get_train_eval_sets()
    if train:
        filenames = train_filenames
    else:
        filenames = eval_filenames
    label_dic = get_label_dic()
    labels = [label_dic[os.path.basename(x)].class_id for x in filenames]

    return filenames, labels

# breaks apart set into train and eval holdout (accomodating batch_size)
def get_train_eval_sets(holdout=HOLDOUT):
    filenames = []
    img_dirs = [os.path.join(FLAGS.train_directory, filename) for filename in
                os.listdir(FLAGS.train_directory)]

    # kaggle sorts images in top level dir by their classes
    for i in range(0, len(img_dirs)):
        # (this is lazy and non-resilient since the order of the dirs listed could be arbitrary but fix it later)
        dir_name = img_dirs[i]
        for filename in [os.path.join(dir_name, x) for x in os.listdir(dir_name)]:
            filename = filename
            filenames.append(filename)
    np.random.shuffle(filenames)

    return filenames[0:NUM_TRAIN_SAMPLES], filenames[-NUM_EVAL_SAMPLES:]
