"""Process audio buffers."""
from multiprocessing import Queue

import numpy as np
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer

from record import RATE

MIN_CONF = 0.25


def process(audio: np.ndarray, analyzer: Analyzer):
    """Process audio buffer.

    Will analyze it using the birdnetlib analyzer.
    """
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    recording = RecordingBuffer(
        analyzer,
        audio,
        RATE,
        min_conf=MIN_CONF,
    )
    recording.analyze()
    return audio, recording.detections


def process_audio(input_queue: Queue, ouput_queue: Queue):
    """Process that continuously processes audio buffers."""
    analyzer = Analyzer()

    while True:
        audio = input_queue.get()
        audio, detections = process(audio, analyzer)
        ouput_queue.put([audio, detections])
