import torch
from logging_config import *

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

logger.info('torch.cuda.is_available() = {} ... using {} as DEVICE ...'.format(torch.cuda.is_available(), DEVICE))
