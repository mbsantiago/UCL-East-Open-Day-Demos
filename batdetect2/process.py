"""Process audio buffers."""
from multiprocessing import Queue

import numpy as np


def process_audio(input_queue: Queue, ouput_queue: Queue):
    """Process that continuously processes audio buffers."""
    from batdetect2 import api

    config = api.get_config(detection_threshold=0.5)

    while True:
        audio = input_queue.get()
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max
        detections, _, _ = api.process_audio(audio, config=config)
        ouput_queue.put([audio, detections])
