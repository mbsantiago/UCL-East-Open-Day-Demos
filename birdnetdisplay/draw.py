"""Draw the spectrogram and detections using matplotlib."""
from multiprocessing import Queue
from queue import Empty

import matplotlib.animation as animation
import matplotlib.pyplot as plt

plt.style.use("dark_background")


def plot(q: Queue):
    """Use matplotlib to plot the spectrogram and detections."""
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(1, 1, 1)

    def animate(_):
        try:
            spec, detections = q.get_nowait()
            ax1.clear()
            ax1.pcolormesh(
                spec,
                cmap="gray",
                vmax=0,
                vmin=-80,
            )

            for num, detection in enumerate(detections):
                ax1.text(
                    0,
                    num * 10,
                    f"{detection['common_name']} {detection['confidence']:.2f}",
                    color="red",
                    fontsize=20,
                )

            ax1.axis("off")
        except Empty:
            pass

    ani = animation.FuncAnimation(fig, animate, interval=50)
    plt.show()
    del ani
