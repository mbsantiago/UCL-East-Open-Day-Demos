"""Pygame visualization for audio and detections."""
from multiprocessing import Queue
from queue import Empty

import librosa
import numpy as np
import pygame
from matplotlib.cm import get_cmap
from record import DURATION, RATE

from batdetect2.detector.parameters import TARGET_SAMPLERATE_HZ


def convert_time_to_pixels(time, width):
    """Convert time in seconds to pixels."""
    target_duration = DURATION * RATE / TARGET_SAMPLERATE_HZ
    return int(time * width / target_duration)


def convert_frequency_to_pixels(freq, height):
    """Convert frequency in Hz to pixels."""
    nyquist = RATE / 2
    audio_freq = freq * RATE / TARGET_SAMPLERATE_HZ
    max = librosa.hz_to_mel(nyquist)
    mel = librosa.hz_to_mel(audio_freq)
    return int((mel / max) * height)


EMPTY = pygame.Color(0, 0, 0, 0)
FONT_SIZE = 40
BAR_SIZE = 400
SPEC_CMAP = "cividis"
PROBABILITY_CMAP = "magma"
SPEC_SHAPE = (259, 128)

TEXT_COLOR = (255, 255, 255)
CONFIDENCE_THRESHOLD = 0.3
REF = 700


class FPS:
    """Class to render FPS on screen."""

    def __init__(self):
        """Initialize FPS class."""
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Verdana", FONT_SIZE)
        self.text = self.font.render(
            str(self.clock.get_fps()),
            True,
            TEXT_COLOR,
        )

    def render(self, display):
        """Render FPS on screen."""
        self.text = self.font.render(
            str(round(self.clock.get_fps(), 2)),
            True,
            TEXT_COLOR,
        )
        display.blit(self.text, (4, 4))


class Spectrogram:
    """Class to render spectrogram on screen."""

    def __init__(self, shape, cmap):
        """Initialize Spectrogram class."""
        self.shape = shape
        self.cmap = cmap
        self.spec_surface = pygame.Surface(self.shape)

    def render(self, audio, display):
        """Render spectrogram on screen."""
        # Compute mel spectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=RATE)
        spec = librosa.power_to_db(spec, ref=REF)

        # Clamp the spec to -80 to 0
        spec = np.flipud(np.clip(spec, -80, 0))
        spec = 255 * self.cmap((spec.T + 80) / 80)[:, :, :3]

        # Draw spec to surface and scale to screen size
        pygame.surfarray.blit_array(self.spec_surface, spec)
        image = pygame.transform.scale(self.spec_surface, display.get_size())

        # Draw spectrogram
        display.blit(image, (0, 0))


def array_to_color(array):
    """Convert array to color."""
    return (255 * np.array(array)).astype(np.uint8)


class Detections:
    """Class to render detections on screen."""

    def __init__(
        self,
        cmap,
        padding=4,
        font_size=FONT_SIZE,
    ):
        """Initialize Detections class."""
        self.cmap = cmap
        self.font = pygame.font.SysFont("Verdana", font_size)
        self.padding = padding

    def render(self, detections, display):
        """Render detections on screen."""
        width = display.get_width()
        height = display.get_height()

        for detection in detections:
            x_start = convert_time_to_pixels(detection["start_time"], width)
            x_end = convert_time_to_pixels(detection["end_time"], width)
            y_low = convert_frequency_to_pixels(detection["low_freq"], height)
            y_high = convert_frequency_to_pixels(
                detection["high_freq"], height
            )
            color = array_to_color(self.cmap(detection["det_prob"]))
            pygame.draw.rect(
                display,
                color,  # type: ignore
                (
                    x_start,
                    height - y_high,
                    x_end - x_start,
                    y_high - y_low,
                ),
                1,
            )

            class_text = self.font.render(
                detection["class"],
                True,
                color,  # type: ignore
            )
            text_height = class_text.get_height()
            display.blit(
                class_text,
                (
                    x_start + self.padding,
                    height - y_high - text_height - self.padding,
                ),
            )

            prob_text = self.font.render(
                f"{detection['det_prob']:.2f}",
                True,
                color,  # type: ignore
            )
            display.blit(
                prob_text,
                (x_start + self.padding, height - y_low + self.padding),
            )


def run_pygame(q: Queue):
    """Run pygame visualization."""
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))

    fps = FPS()

    spec = Spectrogram(SPEC_SHAPE, cmap=get_cmap(SPEC_CMAP))

    detections = Detections(cmap=get_cmap(PROBABILITY_CMAP))

    # Main loop
    running = True
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        try:
            # get the latest data from the queue
            # this will throw an exception if the queue is empty
            # so we need to wrap it in a try/except
            audio, dets = q.get_nowait()

            # Draw spectrogram
            spec.render(audio, screen)

            # Draw detections
            detections.render(dets, screen)

            pygame.display.flip()
        except Empty:
            pass

            fps.clock.tick(60)  # limits FPS to 60

    pygame.quit()
