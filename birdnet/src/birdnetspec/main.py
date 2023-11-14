"""Real time audio display and analysis."""
from multiprocessing import Process, Queue

from .process import process_audio
from .draw import run_pygame
from .record import record


def main():
    """Run the main program."""
    audio_queue1 = Queue(maxsize=1)
    audio_queue2 = Queue(maxsize=1)
    spec_queue = Queue(maxsize=1)

    p_record = Process(target=record, args=(audio_queue1, audio_queue2))
    p_process = Process(target=process_audio, args=(audio_queue1, spec_queue))
    p_plot = Process(target=run_pygame, args=(spec_queue, audio_queue2,))

    p_record.start()
    p_process.start()
    p_plot.start()

    try:
        p_record.join()
        p_process.join()
        p_plot.join()

    except KeyboardInterrupt:
        p_record.terminate()
        p_process.terminate()
        p_plot.terminate()


if __name__ == "__main__":
    main()
