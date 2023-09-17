"""Record audio continuously into a buffer and put it into a queue."""
import time
from datetime import datetime
from multiprocessing import Queue
from queue import Full

import numpy as np
import pyaudio

DELAY = 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
HEIGHT = CHUNK // 2 + 1
WIDTH = 256
DURATION = 3


def record(q1: Queue, q2: Queue):
    """Record audio continuously into a buffer and put it into a queue."""
    audio = np.zeros(
        int((DURATION + DELAY) * RATE), dtype=np.int16
    )
    current = 0

    p = pyaudio.PyAudio()

    def callback(input_data, frame_count, *_):
        nonlocal current, audio

        now = datetime.now()
        data = np.frombuffer(input_data, dtype=np.int16)

        audio = np.roll(audio, -frame_count)
        audio[-frame_count:] = data
        current += frame_count

        try:
            q1.put_nowait([audio, now])
        except Full:
            pass

        try:
            q2.put_nowait([audio, now])
        except Full:
            pass

        return input_data, pyaudio.paContinue

    # start Recording
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        stream_callback=callback,
        frames_per_buffer=CHUNK,
    )

    # Wait for stream to finish (4)
    while stream.is_active():
        time.sleep(0.1)

    # Close the stream (5)
    stream.close()

    # Release PortAudio system resources (6)
    p.terminate()
