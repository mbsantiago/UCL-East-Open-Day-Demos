"""Process audio buffers."""
from multiprocessing import Queue

import numpy as np
from batdetect2 import api


def process(audio: np.ndarray):
    """Process audio buffer.

    Will analyze it using the birdnetlib analyzer.
    """
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    detections, _, _ = api.process_audio(audio)
    return audio, detections


def process_audio(input_queue: Queue, ouput_queue: Queue):
    """Process that continuously processes audio buffers."""
    while True:
        audio = input_queue.get()
        audio, detections = process(audio)
        ouput_queue.put([audio, detections])
