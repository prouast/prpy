###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import sys
sys.path.append('../propy')

from propy.torch.model_saver import Candidate, ModelSaver

import logging
import os
import pytest
import shutil
import torch

# https://stackoverflow.com/questions/40710094/how-to-suppress-py-test-internal-deprecation-warnings

def test_candidate():
  cand = Candidate(score=0.5, dir='testdir', filename='test')
  assert cand.filepath == 'testdir/test'
  assert cand.score == 0.5

@pytest.mark.parametrize("save_optimizer", [False, True])
def test_model_saver(save_optimizer):
  model_saver = ModelSaver(
      dir='checkpoints', keep_best=2, keep_latest=1, save_optimizer=save_optimizer,
      compare_fn=lambda x,y: x.score > y.score, sort_reverse=True,
      log_fn=logging.info)
  # Dummy model
  class ModelClass(torch.nn.Module):
    def __init__(self):
      super(ModelClass, self).__init__()
      self.fc = torch.nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)
  model = ModelClass()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  # Save latest
  info = model, optimizer, 0, None, None
  model_saver.save_latest(info=info, step=0, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 0
  assert os.path.exists('checkpoints/model_latest_0.pt')
  # Save best
  info = model, optimizer, 10, None, None
  model_saver.save_best(info=info, score=0.1, step=10, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 1
  assert os.path.exists('checkpoints/model_best_10.pt')
  # Save latest
  info = model, optimizer, 20, None, None
  model_saver.save_latest(info=info, step=20, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 1
  assert not os.path.exists('checkpoints/model_latest_0.pt')
  assert os.path.exists('checkpoints/model_latest_20.pt')
  # Save best
  info = model, optimizer, 30, None, None
  model_saver.save_best(info=info, score=0.2, step=30, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert os.path.exists('checkpoints/model_best_10.pt')
  assert os.path.exists('checkpoints/model_best_30.pt')
  # Save latest
  info = model, optimizer, 40, None, None
  model_saver.save_latest(info=info, step=40, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert not os.path.exists('checkpoints/model_latest_0.pt')
  assert not os.path.exists('checkpoints/model_latest_20.pt')
  assert os.path.exists('checkpoints/model_latest_40.pt')
  # Save best
  info = model, optimizer, 50, None, None
  model_saver.save_best(info=info, score=0.3, step=50, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert not os.path.exists('checkpoints/model_best_10.pt')
  assert os.path.exists('checkpoints/model_best_30.pt')
  assert os.path.exists('checkpoints/model_best_50.pt')
  # Save keep
  info = model, optimizer, 60, None, None
  model_saver.save_keep(info=info, step=60, name='model')
  assert len(model_saver.latest_candidates) == 1
  assert len(model_saver.best_candidates) == 2
  assert os.path.exists('checkpoints/model_keep_60.pt')
  # Remove files
  shutil.rmtree('checkpoints')
