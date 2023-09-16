"""Pygame visualization for audio and detections."""
from multiprocessing import Queue
from queue import Empty

import librosa
import numpy as np
import pygame
from matplotlib.cm import get_cmap

from record import RATE

EMPTY = pygame.Color(0, 0, 0, 0)
FONT_SIZE = 100
BAR_SIZE = 800
SPEC_CMAP = "cividis"
PROBABILITY_CMAP = "magma"
SPEC_SHAPE = (259, 128)
DETECTIONS_SHAPE = (1800, 1200)
BLACK = (255, 255, 255)
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
            BLACK,
        )

    def render(self, display):
        """Render FPS on screen."""
        self.text = self.font.render(
            str(round(self.clock.get_fps(), 2)),
            True,
            BLACK,
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


class MovingAverage:
    """Class to compute moving average of detections."""

    def __init__(self, alpha=0.5):
        """Initialize MovingAverage class."""
        self._dict = {}
        self.alpha = alpha

    def update(self, detections):
        """Update moving average with new detections."""
        updated_names = set()

        # Update new detections
        for detection in detections:
            class_name = detection["common_name"]
            probability = detection["confidence"]
            self._dict[class_name] = self._dict.get(
                class_name, 0
            ) * self.alpha + probability * (1 - self.alpha)
            updated_names.add(class_name)

        # Decay old detections
        for class_name in set(self._dict.keys()) - updated_names:
            self._dict[class_name] = self._dict[class_name] * self.alpha

        # Remove detections with low probability
        self._dict = {key: value for key, value in self._dict.items() if value > 0.01}

    def get(self):
        """Return sorted list of detections."""
        return sorted(self._dict.items(), key=lambda x: x[1], reverse=True)


class Detections:
    """Class to render detections on screen."""

    def __init__(
        self,
        shape,
        cmap,
        padding=4,
        font_size=FONT_SIZE,
        bar_width=BAR_SIZE,
    ):
        """Initialize Detections class."""
        self.shape = shape
        self.cmap = cmap
        self.surface = pygame.Surface(shape, pygame.SRCALPHA)
        self.mavg = MovingAverage()
        self.font = pygame.font.SysFont("Verdana", font_size)
        self.padding = padding
        self.bar_width = bar_width

    def render(self, detections, display):
        """Render detections on screen."""
        self.mavg.update(detections)
        detections = self.mavg.get()

        self.surface.fill(EMPTY)

        y = self.padding
        for class_name, probability in detections:
            if probability < CONFIDENCE_THRESHOLD:
                break

            color = (255 * np.array(self.cmap(probability))).astype(np.uint8)
            text = self.font.render(
                f"{class_name}: {probability:.2f}",
                True,
                BLACK,
            )

            height = text.get_height()
            pygame.draw.rect(
                self.surface,
                color,  # type: ignore
                (0, y, int(self.bar_width * probability), height + 8),
            )
            self.surface.blit(text, (self.padding, y + self.padding))
            y = y + text.get_height() + 3 * self.padding

        display.blit(self.surface, (0, 0))


def run_pygame(q: Queue):
    """Run pygame visualization."""
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    fps = FPS()

    spec = Spectrogram(SPEC_SHAPE, cmap=get_cmap(SPEC_CMAP))

    detections = Detections(DETECTIONS_SHAPE, cmap=get_cmap(PROBABILITY_CMAP))

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

            # Draw FPS
            # fps.render(screen)

            pygame.display.flip()
        except Empty:
            pass

            fps.clock.tick(60)  # limits FPS to 60

    pygame.quit()
