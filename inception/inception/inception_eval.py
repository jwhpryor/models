# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import csv

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception import kg_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('predict_dir', '/tmp/imagenet_predict',
                           """Directory where to write predictions.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', kg_data.NUM_EVAL_SAMPLES,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

# Flags govering where predictions are written
tf.app.flags.DEFINE_string('prediction_output', '/tmp/imagenet_predict/predictions.csv',
                           """The location where predictions should be written'.""")

def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      count_top_5 = 0.0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        top_1, top_5 = sess.run([top_1_op, top_5_op])
        count_top_1 += np.sum(top_1)
        count_top_5 += np.sum(top_5)
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Compute precision @ 1.
      precision_at_1 = count_top_1 / total_sample_count
      recall_at_5 = count_top_5 / total_sample_count
      print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
            (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
      summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

def top_k(a, N):
  return np.argsort(a)[::-1][:N]

def write_predictions(predictions):
  print('Writing ' + str(len(predictions)) + ' predictions...')
  f = open(FLAGS.prediction_output, 'wt')
  filename_observed_hack = set()
  try:
    writer = csv.writer(f)
    writer.writerow(('img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'))
    for p in predictions:
      filename = p[0]
      logit = p[1]

      # dont know how to do this with numpy properly
      top_idx = top_k(logit, 3)
      logit_mod = [0 for x in logit]
      for i in xrange(len(logit)):
        if i in top_idx:
         logit_mod[i] = logit[i]
      #logit = logit_mod

      # normalize (should do this in tensorflow)
      max = np.max(logit_mod)
      min = np.min(logit_mod)
      range = max - min
      logit_mod = np.subtract(logit_mod, min)
      logit_mod = np.divide(logit_mod, range)

      # because of my batching I might see duplicates this hack avoids
      if not filename in filename_observed_hack:
        filename_observed_hack.add(filename)
      else:
        continue

      #writer.writerow([filename] + logit.tolist())
      writer.writerow([filename] + logit_mod)
  finally:
    f.close()

def _predict_once(saver, filename_op, logits_op):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    # restore checkpoint
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      #num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      num_iter = int(math.ceil(kg_data.NUM_PREDICT_SAMPLES / FLAGS.batch_size))
      step = 0

      # Run predictions
      print('%s: starting prediction on (%s).' % (datetime.now(), FLAGS.subset))
      predictions = []

      ########################################
      #num_iter = 1

      while step < num_iter and not coord.should_stop():
        if step % 10 == 0:
            print('step ' + str(step) + ' of ' + str(num_iter))

        filenames, logits = sess.run([filename_op, logits_op])
        for filename, logit in zip(filenames, logits):
          #print(filename + ":" + str(logit))
          predictions.append((filename, logit))
        step += 1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    try:
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    except Exception as e:
        print('Ignoring e: ' + str(e))

    write_predictions(predictions)

def predict(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels, filenames = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference(images, num_classes)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
      inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    _predict_once(saver, filenames, logits)

def evaluate_op(dataset):
    # Get images and labels from the dataset.
    images, labels, _ = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    #num_classes = dataset.num_classes() + 1
    num_classes = dataset.num_classes()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference(images, num_classes)

    # Calculate predictions.
    top_1_op = tf.nn.in_top_k(logits, labels, 1)
    top_5_op = tf.nn.in_top_k(logits, labels, 5)

    return top_1_op, top_5_op

def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    top_1_op, top_5_op = evaluate_op(dataset)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
      inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, graph_def=graph_def)

    _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op)
