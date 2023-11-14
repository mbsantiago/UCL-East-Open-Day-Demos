"""Process audio buffers."""
from multiprocessing import Queue

import numpy as np
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer

from .record import DELAY, RATE

MIN_CONF = 0.25

delay_length = int(RATE * DELAY)


def process(audio: np.ndarray, analyzer: Analyzer):
    """Process audio buffer.

    Will analyze it using the birdnetlib analyzer.
    """
    audio = audio[delay_length:].astype(np.float32) / np.iinfo(np.int16).max
    recording = RecordingBuffer(
        analyzer,
        audio,
        RATE,
        min_conf=MIN_CONF,
    )
    recording.analyze()
    return recording.detections


def process_audio(input_queue: Queue, ouput_queue: Queue):
    """Process that continuously processes audio buffers."""
    analyzer = Analyzer()

    while True:
        audio, time_info = input_queue.get()
        detections = process(audio, analyzer)
        ouput_queue.put([detections, time_info])
