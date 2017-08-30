# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""ResNet model for classifying images from CIFAR-10 dataset.

Support single-host training with one or multiple devices.

ResNet as proposed in:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. arXiv:1512.03385

CIFAR-10 as in:
http://www.cs.toronto.edu/~kriz/cifar.html


"""
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys

import cifar10
import cifar10_model
import cifar10_utils
import multigpu_estimator

tf.logging.set_verbosity(tf.logging.INFO)

def optimizer_fn(params):
  num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch(
      'train') // (params.train_batch_size * params.num_workers)
  boundaries = [
      num_batches_per_epoch * x
      for x in np.array([82, 123, 300], dtype=np.int64)
  ]
  staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]

  learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                              boundaries, staged_lr)
  # Create a nicely-named tensor for logging
  learning_rate = tf.identity(learning_rate, name='learning_rate')

  return tf.train.MomentumOptimizer(
      learning_rate=learning_rate, momentum=params.momentum)

def model_fn(mode, features, labels, params):
  """Build computation tower for each device (CPU or GPU).

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    tower_losses: a list to be appended with current tower's loss.
    tower_gradvars: a list to be appended with current tower's gradients.
    tower_preds: a list to be appended with current tower's predictions.
    is_cpu: true if build tower on CPU.
  """
  # Workaround because ServingInputReceiver must pass dict
  if isinstance(features, dict):
    features = features['feature']

  if params.variable_strategy == multigpu_estimator.VariableStrategy.CPU:
    data_format = 'channels_last'
  else:
    data_format = 'channels_first'

  model = cifar10_model.ResNetCifar10(
      params.num_layers,
      batch_norm_decay=params.batch_norm_decay,
      batch_norm_epsilon=params.batch_norm_epsilon,
      is_training=bool(mode == ModeKeys.TRAIN),
      data_format=data_format)

  logits = model.forward_pass(features, input_data_format='channels_first')

  if mode in (ModeKeys.EVAL, ModeKeys.TRAIN):
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    loss = tf.reduce_mean(loss, name='loss')

  if mode in (ModeKeys.EVAL, ModeKeys.PREDICT):
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }
    if isinstance(features, dict) and 'keys' in features:
      predictions['keys'] = features['keys']

  if mode == ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'serving_default': tf.estimator.export.PredictOutput(
                outputs=predictions)
        }
    )

  if mode == ModeKeys.EVAL:
    labels_one_hot = tf.one_hot(
        labels,
        depth=predictions['probabilities'].shape[1],
        on_value=True,
        off_value=False,
        dtype=tf.bool
    )
    metrics = {
        'accuracy': tf.metrics.accuracy(labels, predictions['classes']),
        'auroc': tf.metrics.auc(labels_one_hot, predictions['probabilities'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
  if mode == ModeKeys.TRAIN:
    return tf.estimator.EstimatorSpec(
        # Dummy train_op, unused by TowerEstimator
        train_op=loss,
        loss=loss,
        mode=mode
    )


def serving_input_fn():
  with tf.device('/cpu:0'):
    image_bytes_batch = tf.placeholder(shape=[None], dtype=tf.string)

    images = tf.map_fn(
        functools.partial(tf.image.decode_jpeg, channels=cifar10.DEPTH),
        image_bytes_batch,
        dtype=tf.uint8
    )
    resized_images = tf.image.resize_images(
        images, [cifar10.HEIGHT, cifar10.WIDTH])
    # Reshape to [depth, height, width]
    final_images = tf.cast(
        tf.transpose(resized_images, [0, 3, 1, 2]), tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      final_images, image_bytes_batch)


def input_fn(data_dir, subset, batch_size,
             use_distortion_for_training=True):
  """Create input graph for model.

  Args:
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  with tf.device('/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
    return dataset.make_batch(batch_size)

def experiment_fn(run_config, hparams):
  """Returns an Experiment."""
  # Create estimator.
  train_input_fn = functools.partial(
      input_fn,
      hparams.data_dir,
      subset='train',
      batch_size=hparams.train_batch_size,
      use_distortion_for_training=hparams.use_distortion_for_training)

  eval_input_fn = functools.partial(
      input_fn,
      hparams.data_dir,
      subset='eval',
      batch_size=hparams.eval_batch_size)

  num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
  if num_eval_examples % hparams.eval_batch_size != 0:
    raise ValueError('validation set size must be multiple of eval_batch_size')

  train_steps = hparams.train_steps
  eval_steps = num_eval_examples // hparams.eval_batch_size
  examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(
    hparams.train_batch_size, every_n_steps=10)

  tensors_to_log = {'learning_rate': 'learning_rate',
                    'loss': 'loss'}

  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100)

  hooks = [logging_hook, examples_sec_hook]

  classifier = multigpu_estimator.TowerEstimator(
      model_fn=model_fn,
      variable_strategy=hparams.variable_strategy,
      optimizer_fn=optimizer_fn,
      sync_replicas=hparams.sync,
      config=run_config,
      params=hparams)

  # Create experiment.
  experiment = tf.contrib.learn.Experiment(
      classifier,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      export_strategies=tf.contrib.learn.make_export_strategy(
          serving_input_fn,
          default_output_alternative_key=None,
          exports_to_keep=1
      )
  )
  # Adding hooks to be used by the estimator on training modes
  experiment.extend_train_hooks(hooks)
  return experiment


def main(job_dir,
         log_device_placement,
         num_intra_threads,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(
          force_gpu_compatible=True
      )
  )

  config = cifar10_utils.RunConfig(
      session_config=sess_config,
      model_dir=job_dir)
  tf.contrib.learn.learn_runner.run(
      experiment_fn,
      run_config=config,
      hparams=tf.contrib.training.HParams(
          num_workers=config.num_worker_replicas or 1,
          **hparams)
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the CIFAR-10 input data is stored.'
  )
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.'
  )
  parser.add_argument(
      '--variable-strategy',
      choices=[multigpu_estimator.VariableStrategy.CPU,
               multigpu_estimator.VariableStrategy.GPU],
      type=str,
      default=multigpu_estimator.VariableStrategy.CPU,
      help='Where to locate variable operations'
  )
  parser.add_argument(
      '--num-layers',
      type=int,
      default=44,
      help='The number of layers of the model.'
  )
  parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.'
  )
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.'
  )
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.'
  )
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.'
  )
  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.'
  )
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """
  )
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.'
  )
  parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present when running in a distributed environment will run on sync mode.\
      """
  )
  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """
  )
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """
  )
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.'
  )
  parser.add_argument(
      '--batch-norm-decay',
      type=float,
      default=0.997,
      help='Decay for batch norm.'
  )
  parser.add_argument(
      '--batch-norm-epsilon',
      type=float,
      default=1e-5,
      help='Epsilon for batch norm.'
  )
  args = parser.parse_args()

  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid --num-layers parameter.')
  main(**vars(args))
