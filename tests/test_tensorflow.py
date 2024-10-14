# Copyright (c) 2024 Philipp Rouast
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
sys.path.append('../prpy')

import logging
import numpy as np
import os
from packaging import version
import pytest
import shutil
import tensorflow as tf

# TODO: Consider testing for no side effects

# https://stackoverflow.com/questions/40710094/how-to-suppress-py-test-internal-deprecation-warnings

def tf_function_wrapper(func):
  @tf.function
  def tf_function_func(*args, **kwargs):
    return func(*args, **kwargs)
  return tf_function_func

## Signal

from prpy.tensorflow.signal import normalize, scale, standardize, diff

@pytest.mark.parametrize("tf_function", [False, True])
def test_normalize(tf_function):
  normalize_f = tf_function_wrapper(normalize) if tf_function else normalize
  # Check with axis=-1
  tf.debugging.assert_near(
    x=normalize_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=-1),
    y=tf.convert_to_tensor([[0.2, 1.2, -2.8, 2.2, -0.8], [-1, 4, -4, 1, 0]]))
  # Check with axis=0
  tf.debugging.assert_near(
    normalize_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=0),
    tf.convert_to_tensor([[0., -2., 0., 0., -1.], [0., 2., 0., 0., 1.]]))
  # Check with axis=(0,1)
  tf.debugging.assert_near(
    normalize_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=(0,1)),
    tf.convert_to_tensor([[-.4, .6, -3.4, 1.6, -1.4], [-.4, 4.6, -3.4, 1.6, .6]]))

@pytest.mark.parametrize("tf_function", [False, True])
def test_scale(tf_function):
  scale_f = tf_function_wrapper(scale) if tf_function else scale
  # Check with axis=-1
  tf.debugging.assert_near(
    x=scale_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=-1),
    y=tf.convert_to_tensor([[0., 0.5812382, -1.7437146, 1.1624764, -0.5812382],
                            [0., 1.9174124, -1.1504475, 0.766965, 0.3834825]]))
  # Check with axis=0
  tf.debugging.assert_near(
    scale_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=0),
    tf.convert_to_tensor([[0., 0.5, 0., 0., -1.], [0., 2.5, 0., 0., 1.]]))
  # Check with axis=(0,1)
  tf.debugging.assert_near(
    scale_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=(0,1)),
    tf.convert_to_tensor([[0., 0.436852, -1.310556, 0.873704, -0.436852],
                          [0., 2.1842601, -1.310556, 0.873704, 0.436852]]))

@pytest.mark.parametrize("tf_function", [False, True])
def test_standardize(tf_function):
  standardize_f = tf_function_wrapper(standardize) if tf_function else standardize
  # Check with axis=-1
  tf.debugging.assert_near(
    standardize_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=-1),
    tf.convert_to_tensor([[.116247639, .697485834, -1.627466946, 1.278724029, -.464990556],
                          [-.383482495, 1.533929979, -1.533929979, .383482495, 0.]]))
  # Check with axis=0
  tf.debugging.assert_near(
    standardize_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [1., 5., 3., 3., 1.]]), axis=0),
    tf.convert_to_tensor([[-1., -1., -1., -1., -1.], [1., 1., 1., 1., 1.]]))
  # Check with axis=(0,1)
  tf.debugging.assert_near(
    standardize_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=(0,1)),
    tf.convert_to_tensor([[-.174740811, .262111217, -1.485296896, .698963245, -.61159284],
                          [-.174740811, 2.00951933, -1.485296896, .698963245, .262111217]]))

@pytest.mark.parametrize("tf_function", [False, True])
def test_diff(tf_function):
  diff_f = tf_function_wrapper(diff) if tf_function else diff
  # Check with axis=1
  tf.debugging.assert_near(
    diff_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=1),
    tf.convert_to_tensor([[1., -4., 5., -3.], [5., -8, 5., -1.]]))
  # Check with axis=0
  tf.debugging.assert_near(
    diff_f(x=tf.convert_to_tensor([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=0),
    tf.convert_to_tensor([0., 4., 0., 0., 2.]))

## Image

from prpy.tensorflow.image import normalize_images, standardize_images, normalized_image_diff
from prpy.tensorflow.image import resize_with_random_method, random_distortion

@pytest.mark.parametrize("tf_function", [False, True])
def test_normalize_images(tf_function):
  normalize_images_f = tf_function_wrapper(normalize_images) if tf_function else normalize_images
  test_data = tf.convert_to_tensor([[[0., 1., 0.],
                                     [1., 1., 1.],
                                     [0., 1., 0.]],
                                    [[1., 0., 0.],
                                     [0., 0., 0.],
                                     [1., 0., 0.]]])
  # Check with axis=None
  tf.debugging.assert_near(
    normalize_images_f(images=test_data),
    tf.convert_to_tensor([[[-.3888889,  .6111111, -.3888889],
                           [ .6111111,  .6111111,  .6111111],
                           [-.3888889,  .6111111, -.3888889]],
                          [[ .6111111, -.3888889, -.3888889],
                           [-.3888889, -.3888889, -.3888889],
                           [ .6111111, -.3888889, -.3888889]]]))
  # Check with axis=(1, 2) -> normalizing together across axis (1, 2) and separately across axis 0
  tf.debugging.assert_near(
    normalize_images_f(images=test_data, axis=(1, 2)),
    tf.convert_to_tensor([[[-.5555556,   .44444442, -.5555556 ],
                           [ .44444442,  .44444442,  .44444442],
                           [-.5555556,   .44444442, -.5555556]],
                          [[ .7777778,  -.22222222, -.22222222],
                           [-.22222222, -.22222222, -.22222222],
                           [ .7777778,  -.22222222, -.22222222]]]))

@pytest.mark.parametrize("tf_function", [False, True])
def test_standardize_image(tf_function):
  standardize_images_f = tf_function_wrapper(standardize_images) if tf_function else standardize_images
  test_data = tf.convert_to_tensor([[[0., 1., 0.],
                                     [1., 1., 1.],
                                     [0., 1., 0.]],
                                    [[1., 0., 0.],
                                     [0., 0., 0.],
                                     [1., 0., 0.]]])
  # Check with axis=None
  tf.debugging.assert_near(
    standardize_images_f(images=test_data),
    tf.convert_to_tensor([[[-.79772407, 1.2535664, -.79772407],
                           [1.2535664,  1.2535664, 1.2535664],
                           [-.79772407, 1.2535664, -.79772407]],
                          [[1.2535664,  -.79772407, -.79772407],
                           [-.79772407, -.79772407, -.79772407],
                           [1.2535664,  -.79772407, -.79772407]]]))
  # Check with axis=(1, 2) -> normalizing together across axis (1, 2) and separately across axis 0
  tf.debugging.assert_near(
    standardize_images_f(images=test_data, axis=(1, 2)),
    tf.convert_to_tensor([[[-1.118034,   .8944272, -1.118034],
                           [  .8944272,  .8944272,   .8944272],
                           [-1.118034,   .8944272, -1.118034]],
                          [[ 1.8708289, -.53452253, -.53452253],
                           [-.53452253, -.53452253, -.53452253],
                           [ 1.8708289, -.53452253, -.53452253]]]))

@pytest.mark.parametrize("tf_function", [False, True])
def test_normalized_image_diff(tf_function):
  normalized_image_diff_f = tf_function_wrapper(normalized_image_diff) if tf_function else normalized_image_diff
  test_data = tf.convert_to_tensor([[[.1, .2, .1],
                                     [.2, .3, .2],
                                     [.1, .2, .1]],
                                    [[.2, .1, .1],
                                     [.1, .1, .1],
                                     [.2, .1, .1]]])
  # Check with axis=0
  tf.debugging.assert_near(
    normalized_image_diff_f(images=test_data, axis=0),
    tf.convert_to_tensor([[[ .3333333, -.3333333,  0.        ],
                           [-.3333333, -.50000006, -.3333333 ],
                           [ .3333333, -.3333333,  0.        ]]]))

@pytest.mark.parametrize("target_shape", [(8, 8), (5, 5)])
@pytest.mark.parametrize("tf_function", [False, True])
def test_resize_with_random_method(tf_function, target_shape):
  resize_with_random_method_f = tf_function_wrapper(resize_with_random_method) if tf_function else resize_with_random_method
  image_in = tf.ones(shape=(12, 12, 3))
  image_1 = resize_with_random_method_f(image_in, target_shape)
  image_2 = resize_with_random_method_f(image_1, target_shape=(12, 12))
  tf.debugging.assert_near(image_in, image_2)

@pytest.mark.parametrize("tf_function", [False, True])
def test_random_distortion(tf_function):
  random_distortion_f = tf_function_wrapper(random_distortion) if tf_function else random_distortion
  out = random_distortion_f(tf.ones(shape=(12, 12, 3)))
  assert out.shape == (12, 12, 3)

## Model saver

from prpy.tensorflow.model_saver import Candidate, ModelSaver

def test_candidate():
  cand = Candidate(score=0.5, dir='testdir', filename='test')
  assert cand.filepath == os.path.join('testdir', 'test')
  assert cand.score == 0.5

@pytest.mark.parametrize("save_format", ["tf", "h5"])
@pytest.mark.parametrize("save_optimizer", [False, True])
def test_model_saver(save_format, save_optimizer):
  model_saver = ModelSaver(
      dir='checkpoints', keep_best=2, keep_latest=1, save_format=save_format,
      save_optimizer=save_optimizer, compare_fn=lambda x,y: x.score > y.score,
      sort_reverse=True, log_fn=logging.info)
  # Dummy model
  inputs = tf.keras.Input(shape=(2,))
  outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(inputs)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  if save_format == 'h5':
    suffix = '.h5'
  elif save_format == 'tf' and save_optimizer:
    suffix = ''
  else:
    suffix = '.index'
  # Save latest
  model_saver.save_latest(model=model, step=0, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 0
  assert os.path.exists('checkpoints/model_latest_0{}'.format(suffix))
  # Save best
  model_saver.save_best(model=model, score=0.1, step=10, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 1
  assert os.path.exists('checkpoints/model_best_10{}'.format(suffix))
  # Save latest
  model_saver.save_latest(model=model, step=20, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 1
  assert not os.path.exists('checkpoints/model_latest_0{}'.format(suffix))
  assert os.path.exists('checkpoints/model_latest_20{}'.format(suffix))
  # Save best
  model_saver.save_best(model=model, score=0.2, step=30, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert os.path.exists('checkpoints/model_best_10{}'.format(suffix))
  assert os.path.exists('checkpoints/model_best_30{}'.format(suffix))
  # Save latest
  model_saver.save_latest(model=model, step=40, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert not os.path.exists('checkpoints/model_latest_0{}'.format(suffix))
  assert not os.path.exists('checkpoints/model_latest_20{}'.format(suffix))
  assert os.path.exists('checkpoints/model_latest_40{}'.format(suffix))
  # Save best
  model_saver.save_best(model=model, score=0.3, step=50, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert not os.path.exists('checkpoints/model_best_10{}'.format(suffix))
  assert os.path.exists('checkpoints/model_best_30{}'.format(suffix))
  assert os.path.exists('checkpoints/model_best_50{}'.format(suffix))
  # Save keep
  model_saver.save_keep(model=model, step=60, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert os.path.exists('checkpoints/model_keep_60{}'.format(suffix))
  # Remove files
  shutil.rmtree('checkpoints')

## Loss

from prpy.tensorflow.loss import balanced_sample_weights, smooth_l1_loss, mae_loss

@pytest.mark.parametrize("tf_function", [False, True])
def test_smooth_l1_loss(tf_function):
  smooth_l1_loss_f = tf_function_wrapper(smooth_l1_loss) if tf_function else smooth_l1_loss
  # Basic example 1-d
  tf.debugging.assert_near(
    smooth_l1_loss_f(
      y_true=tf.convert_to_tensor([0.1, 0.5, 1.8, 3.2]),
      y_pred=tf.convert_to_tensor([1.1, -2.5, 1.7, 3.2]),
      keepdims=True),
    tf.convert_to_tensor([0.5, 2.5, .005, 0]))
  tf.debugging.assert_near(
    smooth_l1_loss_f(
      y_true=tf.convert_to_tensor([0.1, 0.5, 1.8, 3.2]),
      y_pred=tf.convert_to_tensor([1.1, -2.5, 1.7, 3.2]),
      keepdims=False),
    tf.convert_to_tensor(0.75125))
  # 2-d
  tf.debugging.assert_near(
    smooth_l1_loss_f(
      y_true=tf.convert_to_tensor([[0.1, 0.5, 1.8, 3.2], [-.3, 0.8, 2.1, 1.0]]),
      y_pred=tf.convert_to_tensor([[1.1, -2.5, 1.7, 3.2], [-.6, 0.2, 2.9, 0.0]]),
      keepdims=True),
    tf.convert_to_tensor([[0.5, 2.5, .005, 0], [.045, .18, .32, .5]]))
  tf.debugging.assert_near(
    smooth_l1_loss_f(
      y_true=tf.convert_to_tensor([[0.1, 0.5, 1.8, 3.2], [-.3, 0.8, 2.1, 1.0]]),
      y_pred=tf.convert_to_tensor([[1.1, -2.5, 1.7, 3.2], [-.6, 0.2, 2.9, 0.0]]),
      keepdims=False),
    tf.convert_to_tensor(0.50625))
  # 3-d
  tf.debugging.assert_near(
    smooth_l1_loss_f(
      y_true=tf.convert_to_tensor([[[0.1, 0.5], [1.8, 3.2]], [[-.3, 0.8], [2.1, 1.0]]]),
      y_pred=tf.convert_to_tensor([[[1.1, -2.5], [1.7, 3.2]], [[-.6, 0.2], [2.9, 0.0]]]),
      keepdims=True),
    tf.convert_to_tensor([[[0.5, 2.5], [.005, 0]], [[.045, .18], [.32, .5]]]))
  tf.debugging.assert_near(
    smooth_l1_loss_f(
      y_true=tf.convert_to_tensor([[[0.1, 0.5], [1.8, 3.2]], [[-.3, 0.8], [2.1, 1.0]]]),
      y_pred=tf.convert_to_tensor([[[1.1, -2.5], [1.7, 3.2]], [[-.6, 0.2], [2.9, 0.0]]]),
      keepdims=False),
    tf.convert_to_tensor(0.50625))

@pytest.mark.parametrize("tf_function", [False, True])
def test_mae_loss(tf_function):
  mae_loss_f = tf_function_wrapper(mae_loss) if tf_function else mae_loss
  # Basic example 1-d
  tf.debugging.assert_near(
    mae_loss_f(
      y_true=tf.convert_to_tensor([0.1, 0.5, 1.8, 3.2]),
      y_pred=tf.convert_to_tensor([1.1, -2.5, 1.7, 3.2]),
      keepdims=True),
    tf.convert_to_tensor(1.025))
  tf.debugging.assert_near(
    mae_loss_f(
      y_true=tf.convert_to_tensor([0.1, 0.5, 1.8, 3.2]),
      y_pred=tf.convert_to_tensor([1.1, -2.5, 1.7, 3.2]),
      keepdims=False),
    tf.convert_to_tensor(1.025))
  # 2-d
  tf.debugging.assert_near(
    mae_loss_f(
      y_true=tf.convert_to_tensor([[0.1, 0.5, 1.8, 3.2], [-.3, 0.8, 2.1, 1.0]]),
      y_pred=tf.convert_to_tensor([[1.1, -2.5, 1.7, 3.2], [-.6, 0.2, 2.9, 0.0]]),
      keepdims=True),
    tf.convert_to_tensor([1.025, 0.675]))
  tf.debugging.assert_near(
    mae_loss_f(
      y_true=tf.convert_to_tensor([[0.1, 0.5, 1.8, 3.2], [-.3, 0.8, 2.1, 1.0]]),
      y_pred=tf.convert_to_tensor([[1.1, -2.5, 1.7, 3.2], [-.6, 0.2, 2.9, 0.0]]),
      keepdims=False),
    tf.convert_to_tensor(0.85))
  # 3-d
  tf.debugging.assert_near(
    mae_loss_f(
      y_true=tf.convert_to_tensor([[[0.1, 0.5], [1.8, 3.2]], [[-.3, 0.8], [2.1, 1.0]]]),
      y_pred=tf.convert_to_tensor([[[1.1, -2.5], [1.7, 3.2]], [[-.6, 0.2], [2.9, 0.0]]]),
      keepdims=True),
    tf.convert_to_tensor([[2.0, .05], [.45, .9]]))
  tf.debugging.assert_near(
    mae_loss_f(
      y_true=tf.convert_to_tensor([[[0.1, 0.5], [1.8, 3.2]], [[-.3, 0.8], [2.1, 1.0]]]),
      y_pred=tf.convert_to_tensor([[[1.1, -2.5], [1.7, 3.2]], [[-.6, 0.2], [2.9, 0.0]]]),
      keepdims=False),
    tf.convert_to_tensor(0.85))
  
@pytest.mark.parametrize("tf_function", [False, True])
def test_balanced_sample_weights(tf_function):
  balanced_sample_weights_f = tf_function_wrapper(balanced_sample_weights) if tf_function else balanced_sample_weights
  # Basic use case with shape (5,)
  tf.debugging.assert_near(
    balanced_sample_weights_f(
      labels=tf.convert_to_tensor([1, 1, 2, 4, 1, 2]),
      unique=tf.range(5)
    ),
    tf.convert_to_tensor([0.6666667, 0.6666667, 1., 2., 0.6666667, 1.]))
  # Basic use case with shape (5, 1) 
  tf.debugging.assert_near(
    balanced_sample_weights_f(
      labels=tf.convert_to_tensor([[1], [1], [2], [4], [1], [2]]),
      unique=tf.range(5)
    ),
    tf.convert_to_tensor([[0.6666667], [0.6666667], [1.], [2.], [0.6666667], [1.]]))

## Optimizer

from prpy.tensorflow.optimizer import EpochAdam, EpochAdamW, EpochLossScaleOptimizer

def test_epoch_adam():
  lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[0, 1], values=[0.1, 0.01, 0.001])
  optimizer = EpochAdam(learning_rate=lr_schedule)
  assert optimizer.epochs == 0
  if version.parse(tf.__version__) <= version.parse("2.6.5"):
    assert optimizer._decayed_lr(var_dtype=tf.float32) == 0.1
  else:
    assert optimizer._current_learning_rate == 0.1
  optimizer.finish_epoch()
  assert optimizer.epochs == 1
  if version.parse(tf.__version__) <= version.parse("2.6.5"):
    assert optimizer._decayed_lr(var_dtype=tf.float32) == 0.01
  else:
    assert optimizer._current_learning_rate == 0.01
  optimizer.finish_epoch()
  assert optimizer.epochs == 2
  if version.parse(tf.__version__) <= version.parse("2.6.5"):
    assert optimizer._decayed_lr(var_dtype=tf.float32) == 0.001
  else:
    assert optimizer._current_learning_rate == 0.001

def test_epoch_adam_w():
  lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[0, 1], values=[0.1, 0.01, 0.001])
  optimizer = EpochAdamW(learning_rate=lr_schedule)
  assert optimizer.epochs == 0
  assert optimizer._current_learning_rate == 0.1
  optimizer.finish_epoch()
  assert optimizer.epochs == 1
  assert optimizer._current_learning_rate == 0.01
  optimizer.finish_epoch()
  assert optimizer.epochs == 2
  assert optimizer._current_learning_rate == 0.001

def test_epoch_loss_scale_optimizer():
  lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[0, 1], values=[0.1, 0.01, 0.001])
  optimizer = EpochAdam(learning_rate=lr_schedule)
  ls_optimizer = EpochLossScaleOptimizer(inner_optimizer=optimizer, dynamic=True)
  assert ls_optimizer.epochs == 0
  if version.parse(tf.__version__) <= version.parse("2.6.5"):
    assert ls_optimizer._optimizer._decayed_lr(var_dtype=tf.float32) == 0.1
  else:
    assert ls_optimizer._optimizer._current_learning_rate == 0.1
  ls_optimizer.finish_epoch()
  assert ls_optimizer.epochs == 1
  if version.parse(tf.__version__) <= version.parse("2.6.5"):
    assert ls_optimizer._optimizer._decayed_lr(var_dtype=tf.float32) == 0.01
  else:
    assert ls_optimizer._optimizer._current_learning_rate == 0.01
  ls_optimizer.finish_epoch()
  assert ls_optimizer.epochs == 2
  if version.parse(tf.__version__) <= version.parse("2.6.5"):
    assert ls_optimizer._optimizer._decayed_lr(var_dtype=tf.float32) == 0.001
  else:
    assert ls_optimizer._optimizer._current_learning_rate == 0.001

## LR schedule

from prpy.tensorflow.lr_schedule import PiecewiseConstantDecayWithWarmup

def test_piecewise_constant_decay_with_warmup():
  lr_schedule = PiecewiseConstantDecayWithWarmup(
    boundaries=[9, 19], values=[0.1, 0.01, 0.001], warmup_init_lr=0.01, warmup_steps=3)
  lr_vals = [lr_schedule(i) for i in range(30)]
  tf.debugging.assert_near(
    tf.convert_to_tensor(lr_vals),
    tf.convert_to_tensor([0.01, 0.04, 0.07, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                          0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                          0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]))

## nan

from prpy.tensorflow.nan import reduce_nanmean, reduce_nansum
from prpy.tensorflow.nan import ReduceNanSum, ReduceNanMean, NanLinearCombination

def assert_near_nan(x, y, tol=1e-7):
  nan_mask_x = tf.math.is_nan(x)
  nan_mask_y = tf.math.is_nan(y)
  tf.debugging.assert_equal(nan_mask_x, nan_mask_y)
  if nan_mask_x.get_shape().ndims == 0:
    x = x[tf.newaxis]
    nan_mask_x = nan_mask_x[tf.newaxis]
  if nan_mask_y.get_shape().ndims == 0:
    y = y[tf.newaxis]
    nan_mask_y = nan_mask_y[tf.newaxis] 
  tf.debugging.assert_near(tf.boolean_mask(x, ~nan_mask_x), tf.boolean_mask(y, ~nan_mask_y), atol=tol)

@pytest.mark.parametrize("tf_function", [False, True])
@pytest.mark.parametrize("scenario", [([[1., 1.], [2., np.nan], [np.nan, np.nan]], 4./3., None), # Reduce all
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[1., 2., np.nan], [1., 2., np.nan]], -1), # Reduce one axis
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[1., 2., np.nan]], (0,2)), # Reduce multiple axes
                                      ([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], [[np.nan, np.nan, np.nan]], (0,2))]) # 
def test_reduce_nanmean(tf_function, scenario):
  reduce_nanmean_f = tf_function_wrapper(reduce_nanmean) if tf_function else reduce_nanmean
  x = tf.convert_to_tensor([scenario[0], scenario[0]])
  y = tf.convert_to_tensor(scenario[1])
  # Reduce one axis
  assert_near_nan(
    x=reduce_nanmean_f(x=x, axis=scenario[2]),
    y=y)

@pytest.mark.parametrize("tf_function", [False, True])
@pytest.mark.parametrize("scenario", [([[1., 1.], [2., np.nan], [np.nan, np.nan]], 4./3., [[1./6., 1./6.,], [1./6., 0.], [0., 0.]], None), # Reduce all
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[1., 2., np.nan], [1., 2., np.nan]], [[1./2., 1./2.,], [1., 0.], [0., 0.]], -1), # Reduce one axis
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[1., 2., np.nan]], [[1./4., 1./4.,], [1./2., 0.], [0., 0.]], (0,2)), # Reduce multiple axes
                                      ([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], [[np.nan, np.nan, np.nan]], [[0., 0.], [0., 0.], [0., 0.]], (0,2))]) # 
def test_reduce_nanmean_grad(tf_function, scenario):
  reduce_nanmean_grad = ReduceNanMean(axis=scenario[3])
  reduce_nanmean_grad_f = tf_function_wrapper(reduce_nanmean_grad) if tf_function else reduce_nanmean_grad
  x = tf.convert_to_tensor([scenario[0], scenario[0]])
  y = tf.convert_to_tensor(scenario[1])
  g = tf.convert_to_tensor([scenario[2], scenario[2]])
  # Compute
  with tf.GradientTape() as tape:
    tape.watch(x)
    out = reduce_nanmean_grad_f(x)
  # Check outputs
  assert_near_nan(x=out, y=y)
  # Check gradients
  gradients = tape.gradient(out, x)
  # Perform assertions for the gradients
  tf.debugging.assert_near(x=gradients, y=g)

@pytest.mark.parametrize("tf_function", [False, True])
@pytest.mark.parametrize("scenario", [([[1., 1.], [2., np.nan], [np.nan, np.nan]], 8., None, None, float('nan')), # Reduce all
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], 12., [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], None, float('nan')), # Reduce all with mask
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[2., 2., 0.], [2., 2., 0.]], None, -1, 0), # Reduce one axis
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[2., 4., np.nan], [2., 4., np.nan]], [[1., 1.], [2., 2.], [1., 1.]], -1, float('nan')), # Reduce one axis with mask that needs to be broadcast
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[4., 4., np.nan]], None, (0,2), float('nan')), # Reduce multiple axes
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[4., 8., np.nan]], [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], (0,2), float('nan')), # Reduce multiple axes with mask
                                      ([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], [[np.nan, np.nan, np.nan]], None, (0,2), float('nan'))]) # 
def test_reduce_nansum(tf_function, scenario):
  reduce_nansum_f = tf_function_wrapper(reduce_nansum) if tf_function else reduce_nansum
  x = tf.convert_to_tensor([scenario[0], scenario[0]])
  y = tf.convert_to_tensor(scenario[1])
  assert_near_nan(
    x=reduce_nansum_f(x=x, weight=scenario[2], axis=scenario[3], default=scenario[4]),
    y=y)

@pytest.mark.parametrize("tf_function", [False, True])
@pytest.mark.parametrize("scenario", [([[1., 1.], [2., np.nan], [np.nan, np.nan]], 8., [[1., 1.,], [1., 0.], [0., 0.]], None, None, float('nan')), # Reduce all
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], 12., [[1., 1.,], [1., 0.], [0., 0.]], [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], None, float('nan')), # Reduce all with mask
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[2., 2., 0.], [2., 2., 0.]], [[1., 1.,], [1., 0.], [0., 0.]], None, -1, 0), # Reduce one axis
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[2., 4., np.nan], [2., 4., np.nan]], [[1., 1.,], [1., 0.], [0., 0.]], [[1., 1.], [2., 2.], [1., 1.]], -1, float('nan')), # Reduce one axis with mask that needs to be broadcast
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[4., 4., np.nan]], [[1., 1.,], [1., 0.], [0., 0.]], None, (0,2), float('nan')), # Reduce multiple axes
                                      ([[1., 1.], [2., np.nan], [np.nan, np.nan]], [[4., 8., np.nan]], [[1., 1.,], [1., 0.], [0., 0.]], [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], (0,2), float('nan')), # Reduce multiple axes with mask
                                      ([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]], [[np.nan, np.nan, np.nan]], [[0., 0.,], [0., 0.], [0., 0.]], None, (0,2), float('nan'))]) # 
def test_reduce_nansum_grad(tf_function, scenario):
  reduce_nansum_grad = ReduceNanSum(weight=scenario[3], axis=scenario[4], default=scenario[5])
  reduce_nansum_grad_f = tf_function_wrapper(reduce_nansum_grad) if tf_function else reduce_nansum_grad
  x = tf.convert_to_tensor([scenario[0], scenario[0]])
  y = tf.convert_to_tensor(scenario[1])
  g = tf.convert_to_tensor([scenario[2], scenario[2]])
  # Compute
  with tf.GradientTape() as tape:
    tape.watch(x)
    out = reduce_nansum_grad_f(x)
  # Check outputs
  assert_near_nan(x=out, y=y)
  # Check gradients
  gradients = tape.gradient(out, x)
  # Perform assertions for the gradients
  tf.debugging.assert_near(x=gradients, y=g)

@pytest.mark.parametrize("tf_function", [False, True])
@pytest.mark.parametrize("scenario", [([.4, .7, 1.], [0.], [1., 2., .5], [.4, 1.4, .5], [1., 2., .5], [.6, .3, 0.], [.4, .7, 1.]), # 
                                      ([.4, .7, 1.], [0., 1., 1.], [1., 2., .5], [.4, 1.7, .5], [1., 1., -.5], [.6, .3, 0.], [.4, .7, 1.]),
                                      ([.4, .7, 1.], [0., np.nan, 1.], [1., 2., .5], [.4, np.nan, .5], [1., 0., -.5], [.6, .3, 0.], [.4, .7, 1.])])
def test_nan_linear_combination(tf_function, scenario):
  nan_linear_combination = NanLinearCombination()
  nan_linear_combination_f = tf_function_wrapper(nan_linear_combination) if tf_function else nan_linear_combination
  x = tf.convert_to_tensor(scenario[0])
  val_1 = tf.broadcast_to(tf.convert_to_tensor(scenario[1]), x.shape)
  val_2 = tf.broadcast_to(tf.convert_to_tensor(scenario[2]), x.shape)
  y = tf.convert_to_tensor(scenario[3])
  g_x = tf.convert_to_tensor(scenario[4])
  g_val_1 = tf.convert_to_tensor(scenario[5])
  g_val_2 = tf.convert_to_tensor(scenario[6])
  # Compute
  with tf.GradientTape() as tape:
    tape.watch([x, val_1, val_2])
    out = nan_linear_combination_f(x, val_1, val_2)
  # Check outputs
  assert_near_nan(x=out, y=y)
  # Check gradients
  gradients = tape.gradient(out, [x, val_1, val_2])
  # Perform assertions for the gradients
  tf.debugging.assert_near(x=gradients[0], y=g_x)
  tf.debugging.assert_near(x=gradients[1], y=g_val_1)
  tf.debugging.assert_near(x=gradients[2], y=g_val_2)
  