import logging.config
from logging.config import dictConfig

logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging_config = dict(
    version=1,
    formatters={
        'f': {'format': '[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s'}
    },
    handlers={
        'stream_handler': {'class': 'logging.StreamHandler',
                           'formatter': 'f',
                           'level': logging.DEBUG},
        'file_handler': {'class': 'logging.FileHandler',
                         'formatter': 'f',
                         'level': logging.DEBUG,
                         'filename': 'car-detection.log'
                         }
    },
    root={
        'handlers': ['stream_handler', 'file_handler'],
        'level': logging.DEBUG,
    },
)

dictConfig(logging_config)
