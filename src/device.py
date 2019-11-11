import torch
from logging_config import *

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

logger.info('************ Using {} as device **************'.format(DEVICE))
