###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

"""Save the best models"""

import glob
import logging
import os
import shutil

class Candidate(object):
  """A candidate model with a score"""
  def __init__(self, score, dir, filename):
    self.score = score
    self.filepath = os.path.join(dir, filename)

class ModelSaver(object):
  """Save the best models to disk"""
  def __init__(self, dir="checkpoints", keep_best=5, keep_latest=1, save_format="tf", save_optimizer=False, compare_fn=lambda x,y: x.score < y.score, sort_reverse=True, log_fn=logging.info):
    """Init the ModelSaver"""
    self.best_candidates = []
    self.latest_candidates = []
    # The destination directory (make if necessary)
    self.dir = dir
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)
    self.keep_best = keep_best
    self.keep_latest = keep_latest
    self.save_format = save_format
    self.save_optimizer = save_optimizer
    self.compare_fn_best = compare_fn
    self.compare_fn_latest = lambda x,y: x.score > y.score
    self.sort_reverse = sort_reverse
    self.log_fn = log_fn

  def __save(self, model, filepath):
    if self.save_format == 'h5': filepath += '.h5'
    # Save model
    if self.save_optimizer:
      model.save(
        filepath=filepath, overwrite=True, include_optimizer=True,
        save_format=self.save_format)
    else:
      model.save_weights(
        filepath=filepath, overwrite=True, save_format=self.save_format)

  def save_keep(self, model, step, name):
    self.log_fn("Saving and keeping model for step {}".format(step))
    filepath = os.path.join(self.dir, name + "_keep_" + str(step))
    self.__save(model=model, filepath=filepath)

  def save_latest(self, model, step, name):
    name = name + "_latest_" + str(step)
    # Use step as score
    candidate = Candidate(score=step, dir=self.dir, filename=name)
    if len(self.latest_candidates) < self.keep_latest \
      or self.compare_fn_latest(candidate, self.latest_candidates[-1]):
      self.log_fn("Saving latest model for step {}".format(step))
      # Keep candidate
      self.latest_candidates.append(candidate)
      self.latest_candidates = sorted(
        self.latest_candidates, key=lambda x: x.score, reverse=True)
      # Save candidate
      self.__save(model, filepath=candidate.filepath)
      # Prune candidate
      for candidate in self.latest_candidates[self.keep_latest:]:
        for file in glob.glob(r'{}*'.format(candidate.filepath)):
          if self.save_format == 'tf' and self.save_optimizer:
            shutil.rmtree(file)
          else:
            os.remove(file)
      self.latest_candidates = self.latest_candidates[0:self.keep_latest]

  def save_best(self, model, score, step, name):
    self.log_fn('Saving model for step {}'.format(step))
    name = name + "_best_" + str(step)
    candidate = Candidate(score, dir=self.dir, filename=name)
    if len(self.best_candidates) < self.keep_best \
      or self.compare_fn_best(candidate, self.best_candidates[-1]):
      # Keep candidate
      self.log_fn("Keeping model {} with score {:.4f}".format(
        candidate.filepath, candidate.score))
      self.best_candidates.append(candidate)
      self.best_candidates = sorted(
        self.best_candidates, key=lambda x: x.score, reverse=self.sort_reverse)
      # Save candidate
      self.__save(model, filepath=candidate.filepath)
      # Prune candidates
      for candidate in self.best_candidates[self.keep_best:]:
        self.log_fn('Removing old model {} with score {:.4f}'.format(
          candidate.filepath, candidate.score))
        for file in glob.glob(r'{}*'.format(candidate.filepath)):
          if self.save_format == 'tf' and self.save_optimizer:
            shutil.rmtree(file)
          else:
            os.remove(file)
      self.best_candidates = self.best_candidates[0:self.keep_best]
    else:
      # Skip the candidate
      self.log_fn('Skipping candidate {}'.format(candidate.filepath))
