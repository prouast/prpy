###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import numpy as np
import random as rand
import pytest

@pytest.fixture
def random():
  rand.seed(0)
  np.random.seed(0)
  