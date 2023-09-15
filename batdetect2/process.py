"""Process audio buffers."""
from multiprocessing import Queue

import numpy as np


def process(audio: np.ndarray, api):
    """Process audio buffer.

    Will analyze it using the birdnetlib analyzer.
    """
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    detections, _, _ = api.process_audio(audio)
    detections = []
    return audio, detections


def process_audio(input_queue: Queue, ouput_queue: Queue):
    """Process that continuously processes audio buffers."""
    from batdetect2 import api

    while True:
        audio = input_queue.get()
        audio, detections = process(audio, api)
        ouput_queue.put([audio, detections])
