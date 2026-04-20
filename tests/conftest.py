import random
import pytest
import torch

import numpy as np

TEST_SEED = 1337


@pytest.fixture(autouse=True)
def reproducible_test_environment():
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TEST_SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
