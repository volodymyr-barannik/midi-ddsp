#  Copyright 2022 The MIDI-DDSP Authors.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Train expression generator module."""

import tensorflow as tf
import numpy as np
import time
import logging
import os
import sys
import copy
import argparse

from midi_ddsp.utils.file_utils import pickle_load
from midi_ddsp.utils.training_utils import set_seed
from midi_ddsp.modules.expression_generator_dataset \
  import get_lang_model_dataset
from midi_ddsp.modules.expression_generator import ExpressionGenerator, \
  get_fake_data_expression_generator

set_seed(1111)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
parser = argparse.ArgumentParser(description='Train expression generator.')


def mse_loss(target, pred):
  if isinstance(pred, dict):
    return tf.reduce_mean(tf.keras.losses.MSE(target, pred['raw_output']))
  else:
    return tf.reduce_mean(tf.keras.losses.MSE(target, pred))


def data_aug(batch):
  batch = copy.deepcopy(batch)
  zero_pitch_mask = tf.cast(batch['cond']['note_pitch'] != 0, tf.int64)
  transpose_pitch = np.random.randint(-3, 4)
  batch['cond']['note_pitch'] += transpose_pitch
  batch['cond']['note_pitch'] *= zero_pitch_mask
  time_stretch_rate = np.random.choice([0.9, 0.95, 1, 1.05, 1.1])
  batch['cond']['note_length'] *= time_stretch_rate
  return batch


def train(dataset, total_steps, start_step=1):
  start_time = time.time()
  for step in range(start_step, total_steps + start_step):
    data = next(dataset)
    data = data_aug(data)

    with tf.GradientTape() as tape:
      outputs = model(data['cond'], out=data['target'], training=True)
      loss = loss_fn(data['target'], outputs)

    tf.summary.scalar('Train/loss', loss, step)
    metrics(loss)

    # Clip and apply gradients.
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if step % 100 == 0:
      elapsed = time.time() - start_time
      current_lr = optimizer._decayed_lr('float32').numpy()
      msg = f'| {step:6d} steps | lr {current_lr:02.2e} ' \
            f'| ms/batch {(elapsed * 1000 / 100):5.2f} | ' \
            f'loss: {metrics.result():5.4f}'
      logging.info(msg)
      start_time = time.time()

    if step % 500 == 0:
      evaluate(evaluation_data, step)

    if step % 500 == 0:
      model.save_weights(f'{log_dir}/{step}')


def evaluate(dataset, step):
  eval_metrics = tf.keras.metrics.Mean(name='eval_mean_loss')
  eval_metrics_mse = tf.keras.metrics.Mean(name='eval_mean_loss_mse')
  start_time = time.time()
  for data in dataset:
    outputs = model(data['cond'], out=None, training=False)
    loss = loss_fn(data['target'], outputs)
    eval_metrics(loss)
    loss_mse = mse_loss(data['target'], outputs['output'])
    eval_metrics_mse(loss_mse)

  msg = f'eval: | step {step:6d} | ' \
        f'eval time: {(time.time() - start_time):3.3f} | ' \
        f'loss: {eval_metrics.result():5.4f} | ' \
        f'mse: {eval_metrics_mse.result():5.4f}'
  logging.info(msg)
  tf.summary.scalar('Eval/loss', eval_metrics.result(), step)


if __name__ == '__main__':
  parser.add_argument('--training_epochs', type=int, default=20000,
                      help='Number of training steps.')
  parser.add_argument('--nhid', type=int, default=128,
                      help='Number of hidden units.')
  parser.add_argument('--name', type=str, default='logs_expression_generator',
                      help='Name of the logging directory.')
  parser.add_argument('--batch_size', type=int, default=256,
                      help='Number of batch size.')
  parser.add_argument('--training_set_path', type=str,
                      default=None,
                      help='Path to the training set pickle file.')
  parser.add_argument('--test_set_path', type=str,
                      default=None,
                      help='Path to the test set pickle file.')
  parser.add_argument('--restore_path', type=str,
                      default=None,
                      help='Path to expression generator checkpoint file '
                           'to be restored.')

  args = parser.parse_args()
  training_epochs = args.training_epochs
  train_path = args.training_set_path
  test_path = args.test_set_path
  restore_path = args.restore_path
  n_out = 6
  nhid = args.nhid
  batch_size = args.batch_size
  loss_fn = mse_loss

  log_dir = f'logs/{args.name}'

  model = ExpressionGenerator(n_out=n_out, nhid=nhid)
  _data = get_fake_data_expression_generator(n_out)
  _ = model(_data['cond'], out=_data['target'], training=True)

  scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.99)
  optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)
  metrics = tf.keras.metrics.Mean(name='mean_loss')

  if restore_path:
    model.load_weights(restore_path)

  writer = tf.summary.create_file_writer(log_dir)

  log_path = os.path.join(log_dir, 'train.log')

  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s: %(message)s',
                      handlers=[
                        logging.FileHandler(log_path),
                        logging.StreamHandler(sys.stdout)]
                      )

  model.summary(print_fn=logging.info)

  training_data = get_lang_model_dataset(pickle_load(train_path), repeats=-1,
                                         batch_size=batch_size)
  evaluation_data = get_lang_model_dataset(pickle_load(test_path), repeats=1,
                                           batch_size=batch_size * 3)
  training_data = iter(training_data)

  train(training_data, training_epochs, start_step=1)
