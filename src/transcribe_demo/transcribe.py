from __future__ import annotations

import queue
import threading
from collections.abc import Generator

from transcribe_demo import backend_protocol


def transcribe(
    *,
    backend: backend_protocol.TranscriptionBackend,
    audio_source: backend_protocol.AudioSource,
) -> Generator[backend_protocol.TranscriptionChunk, None, backend_protocol.TranscriptionResult]:
    """
    Generator that yields transcription chunks from the specified backend.

    This function orchestrates the transcription process by:
    1. Running the backend in a background thread
    2. Yielding chunks as they are produced
    3. Returning the final transcription result

    Args:
        backend: Transcription backend implementing TranscriptionBackend protocol
        audio_source: Audio source implementing AudioSource protocol

    Yields:
        TranscriptionChunk: Individual transcription chunks with metadata

    Returns:
        TranscriptionResult: Final transcription result from the backend

    Example:
        backend = WhisperBackend(model_name="turbo", language="en")
        audio_source = FileAudioSource("audio.mp3", sample_rate=16000)

        for chunk in transcribe(backend, audio_source):
            print(chunk.text)
    """
    chunk_queue: queue.Queue[backend_protocol.TranscriptionChunk | None] = queue.Queue()
    result_container: list[backend_protocol.TranscriptionResult] = []
    error_container: list[BaseException] = []

    def backend_worker() -> None:
        """Worker thread that runs the backend and puts chunks in the queue."""
        try:
            result = backend.run(audio_source=audio_source, chunk_queue=chunk_queue)
            result_container.append(result)
        except BaseException as e:
            error_container.append(e)
        finally:
            # Always send sentinel to unblock the generator
            chunk_queue.put(None)

    # Start backend worker thread
    worker_thread = threading.Thread(target=backend_worker, daemon=True)
    worker_thread.start()

    # Yield chunks as they arrive
    try:
        while True:
            chunk = chunk_queue.get()
            if chunk is None:  # Sentinel value
                break
            yield chunk
    finally:
        # Wait for worker to complete
        worker_thread.join()

    # Check for errors
    if error_container:
        raise error_container[0]

    # Return final result
    if not result_container:
        raise RuntimeError("Backend worker completed without producing a result")
    return result_container[0]
