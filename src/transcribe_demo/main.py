from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, TextIO

from transcribe_demo.realtime_backend import run_realtime_transcriber
from transcribe_demo.whisper_backend import (
    TranscriptionChunk,
    run_whisper_transcriber,
)


REALTIME_CHUNK_DURATION = 2.0


class ChunkCollectorWithStitching:
    """
    Collects transcription chunks and displays them.
    With VAD-based chunking, chunks are simply concatenated (no overlap).
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._last_time = float("-inf")
        self._chunks: list[TranscriptionChunk] = []

    @staticmethod
    def _clean_chunk_text(text: str, is_final_chunk: bool = False) -> str:
        """
        Clean trailing punctuation from chunk text for better concatenation.

        Whisper adds punctuation as if each chunk is a complete sentence,
        but VAD splits can occur mid-sentence. We strip trailing commas and
        periods (but keep ? and ! as they're more intentional).
        """
        text = text.strip()
        if not is_final_chunk and text:
            # Strip trailing commas and periods, but not question marks or exclamation points
            while text and text[-1] in '.,':
                text = text[:-1].rstrip()
        return text

    def __call__(
        self,
        chunk_index: int,
        text: str,
        absolute_start: float,
        absolute_end: float,
        inference_seconds: Optional[float] = None,
    ) -> None:
        if not text:
            return

        # Store the chunk
        chunk = TranscriptionChunk(
            index=chunk_index,
            text=text,
            start_time=absolute_start,
            end_time=absolute_end,
            overlap_start=max(0.0, self._last_time),
            inference_seconds=inference_seconds,
        )
        self._chunks.append(chunk)

        # TODO: Sliding window refinement feature (--refine-with-context)
        # When enabled, use a 3-chunk sliding window to refine the middle chunk:
        # 1. Store raw audio buffers for the last 3 chunks
        # 2. When chunk N arrives, concatenate audio from chunks N-2, N-1, N
        # 3. Re-transcribe the concatenated audio with Whisper
        # 4. Use word-level timestamps to extract refined text for chunk N-1 (middle)
        # 5. Display refined version of chunk N-1 after chunk N processing
        #
        # Benefits:
        # - Better context reduces boundary errors
        # - Improved accuracy for cross-chunk phrases
        # - More natural linguistic flow
        #
        # Considerations:
        # - Adds 1-chunk latency (chunk N-1 displayed after N arrives)
        # - Requires ~3x inference time per chunk
        # - Needs word timestamps (not available on MPS/Apple Metal)
        # - Requires storing raw audio buffers
        #
        # Implementation notes:
        # - Add audio buffer storage to whisper_backend.py
        # - Pass raw audio along with transcription in chunk_consumer
        # - Only refine chunks >= 2 (need 3-chunk window)
        # - Extract middle chunk text using word timestamps that fall within N-1 time range
        # - Display both immediate (chunk N) and refined (chunk N-1) with different labels

        # Display the individual chunk
        if inference_seconds is not None:
            # Whisper mode: show actual audio duration and inference time
            chunk_audio_duration = absolute_end - absolute_start
            timing_suffix = f" | audio: {chunk_audio_duration:.2f}s | inference: {inference_seconds:.2f}s"
            label = f"[chunk {chunk_index:03d}{timing_suffix}]"
        else:
            # Realtime mode: show absolute timestamp from session start
            timing_suffix = f" | t={absolute_end:.2f}s"
            label = f"[chunk {chunk_index:03d}{timing_suffix}]"
        use_color = bool(getattr(self._stream, "isatty", lambda: False)())

        if use_color:
            cyan = "\x1b[36m"
            green = "\x1b[32m"
            reset = "\x1b[0m"
            bold = "\x1b[1m"
            label_colored = f"{bold}{cyan}{label}{reset}"
            line = f"{label_colored} {text.strip()}"
        else:
            line = f"{label} {text.strip()}"

        self._stream.write(line + "\n")
        self._stream.flush()
        self._last_time = max(self._last_time, absolute_end)

        # Show concatenated result every few chunks
        if (chunk_index + 1) % 3 == 0:
            # Clean trailing punctuation from all chunks except the last one
            cleaned_chunks = [
                self._clean_chunk_text(c.text, is_final_chunk=(i == len(self._chunks) - 1))
                for i, c in enumerate(self._chunks)
            ]
            concatenated = " ".join(chunk for chunk in cleaned_chunks if chunk)

            if use_color:
                concat_label = f"\n{bold}{green}[CONCATENATED]{reset}"
            else:
                concat_label = f"\n[CONCATENATED]"
            self._stream.write(f"{concat_label} {concatenated}\n\n")
            self._stream.flush()

    def get_final_stitched(self) -> str:
        """Get the final concatenated transcription of all chunks."""
        # Clean trailing punctuation from all chunks except the last one
        cleaned_chunks = [
            self._clean_chunk_text(c.text, is_final_chunk=(i == len(self._chunks) - 1))
            for i, c in enumerate(self._chunks)
        ]
        return " ".join(chunk for chunk in cleaned_chunks if chunk)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream audio from the microphone and transcribe with Whisper or the OpenAI Realtime API."
    )
    parser.add_argument(
        "--backend",
        choices=("whisper", "realtime"),
        default="whisper",
        help="Transcription backend to use.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key for realtime transcription. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default="turbo",
        help="Whisper checkpoint to load (e.g., turbo, tiny.en, tiny, base.en, small).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device to run Whisper on. 'auto' prefers CUDA, then MPS, otherwise CPU.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Input sample rate expected by the model.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of microphone input channels.",
    )
    parser.add_argument(
        "--temp-file",
        type=Path,
        default=None,
        help="Optional path to persist audio chunks for inspection.",
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        default=None,
        help="Custom certificate bundle to trust when downloading Whisper models.",
    )
    parser.add_argument(
        "--insecure-downloads",
        action="store_true",
        help="Disable SSL verification when downloading models (not recommended).",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit immediately if CUDA is unavailable instead of falling back to CPU.",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness level: 0=least aggressive, 3=most aggressive (default: 2).",
    )
    parser.add_argument(
        "--vad-min-silence-duration",
        type=float,
        default=0.2,
        help="Minimum duration of silence (seconds) to trigger chunk split (default: 0.2).",
    )
    parser.add_argument(
        "--vad-min-speech-duration",
        type=float,
        default=0.25,
        help="Minimum duration of speech (seconds) required before transcribing (default: 0.25).",
    )
    parser.add_argument(
        "--vad-speech-pad-duration",
        type=float,
        default=0.2,
        help="Padding duration (seconds) added before speech to avoid cutting words (default: 0.2).",
    )
    parser.add_argument(
        "--max-chunk-duration",
        type=float,
        default=60.0,
        help="Maximum chunk duration in seconds when using VAD (default: 60.0).",
    )
    parser.add_argument(
        "--refine-with-context",
        action="store_true",
        help="[NOT YET IMPLEMENTED] Use 3-chunk sliding window to refine middle chunk transcription. "
             "Improves accuracy with more context but adds 1-chunk latency and ~3x inference time. "
             "Requires word timestamps (not available on MPS/Apple Metal).",
    )
    parser.add_argument(
        "--realtime-model",
        default="gpt-realtime-mini",
        help="Realtime model to use with the OpenAI Realtime API.",
    )
    parser.add_argument(
        "--realtime-endpoint",
        default="wss://api.openai.com/v1/realtime",
        help="Realtime websocket endpoint (advanced).",
    )
    parser.add_argument(
        "--realtime-instructions",
        default=(
            "You are a high-accuracy transcription service. "
            "Return a concise verbatim transcript of the most recent audio buffer. "
            "Do not add commentary or speaker labels."
        ),
        help="Instruction prompt sent to the realtime model.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    # Check for unimplemented features
    if hasattr(args, 'refine_with_context') and args.refine_with_context:
        print(
            "ERROR: --refine-with-context is not yet implemented.\n"
            "This feature will use a 3-chunk sliding window to refine transcriptions with more context.\n"
            "See TODO comments in main.py for implementation details.",
            file=sys.stderr
        )
        sys.exit(1)

    if args.backend == "whisper":
        collector = ChunkCollectorWithStitching(sys.stdout)
        try:
            run_whisper_transcriber(
                model_name=args.model,
                sample_rate=args.samplerate,
                channels=args.channels,
                temp_file=args.temp_file,
                ca_cert=args.ca_cert,
                insecure_downloads=args.insecure_downloads,
                device_preference=args.device,
                require_gpu=args.require_gpu,
                chunk_consumer=collector,
                vad_aggressiveness=args.vad_aggressiveness,
                vad_min_silence_duration=args.vad_min_silence_duration,
                vad_min_speech_duration=args.vad_min_speech_duration,
                vad_speech_pad_duration=args.vad_speech_pad_duration,
                max_chunk_duration=args.max_chunk_duration,
            )
        finally:
            # Show final concatenated result
            final = collector.get_final_stitched()
            if final:
                use_color = sys.stdout.isatty()
                if use_color:
                    green = "\x1b[32m"
                    reset = "\x1b[0m"
                    bold = "\x1b[1m"
                    print(f"\n{bold}{green}[FINAL CONCATENATED]{reset} {final}\n", file=sys.stdout)
                else:
                    print(f"\n[FINAL CONCATENATED] {final}\n", file=sys.stdout)
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key required for realtime transcription. Provide --api-key or set OPENAI_API_KEY."
        )
    collector = ChunkCollectorWithStitching(sys.stdout)
    try:
        run_realtime_transcriber(
            api_key=api_key,
            endpoint=args.realtime_endpoint,
            model=args.realtime_model,
            sample_rate=args.samplerate,
            channels=args.channels,
            chunk_duration=REALTIME_CHUNK_DURATION,
            instructions=args.realtime_instructions,
            insecure_downloads=args.insecure_downloads,
            chunk_consumer=collector,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # Show final concatenated result
        final = collector.get_final_stitched()
        if final:
            use_color = sys.stdout.isatty()
            if use_color:
                green = "\x1b[32m"
                reset = "\x1b[0m"
                bold = "\x1b[1m"
                print(f"\n{bold}{green}[FINAL CONCATENATED]{reset} {final}\n", file=sys.stdout)
            else:
                print(f"\n[FINAL CONCATENATED] {final}\n", file=sys.stdout)


if __name__ == "__main__":
    main()
