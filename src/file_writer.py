import os
from logging_config import *

logger = logging.getLogger(__name__)


class FileWriter(object):
    def __init__(self, file_path, flush_every=5):
        assert os.path.exists(file_path) == False, "File {} already exists. Remove it first".format(file_path)
        self.file_path = file_path
        self.actions = 0
        self.flush_every = flush_every

    def __enter__(self):
        self.fd = open(self.file_path, 'w')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fd.close()

    def append(self, line):
        self.fd.write(line + '\n')
        self.actions += 1
        if self.actions % self.flush_every == 0:
            self.fd.flush()
