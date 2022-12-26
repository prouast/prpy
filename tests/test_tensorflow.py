###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import sys
sys.path.append('../propy')

from propy.tensorflow.signal import normalize, standardize, diff
from propy.tensorflow.image import normalize_images, standardize_images, normalized_image_diff
from propy.tensorflow.image import resize_with_random_method, random_distortion
from propy.tensorflow.model_saver import Candidate, ModelSaver

import logging
import os
import pytest
import shutil
import tensorflow as tf

# https://stackoverflow.com/questions/40710094/how-to-suppress-py-test-internal-deprecation-warnings

def tf_function_wrapper(func):
  @tf.function
  def tf_function_func(*args, **kwargs):
    return func(*args, **kwargs)
  return tf_function_func

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

def test_candidate():
  cand = Candidate(score=0.5, dir='testdir', filename='test')
  assert cand.filepath == 'testdir/test'
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
