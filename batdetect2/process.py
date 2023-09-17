"""Process audio buffers."""
from multiprocessing import Queue

import numpy as np
from record import DELAY, RATE

delay_samples = int(DELAY * RATE)


def process_audio(input_queue: Queue, ouput_queue: Queue):
    """Process that continuously processes audio buffers."""
    from batdetect2 import api

    config = api.get_config(detection_threshold=0.5)

    while True:
        audio, time_info = input_queue.get()
        audio = (
            audio[delay_samples:].astype(np.float32) / np.iinfo(np.int16).max
        )
        detections, _, _ = api.process_audio(audio, config=config)
        ouput_queue.put([detections, time_info])
