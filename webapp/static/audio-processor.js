/**
 * AudioWorklet processor for capturing and converting audio data.
 * Runs in a separate thread from the main UI, preventing stuttering.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.isRecording = false;

        // Listen for control messages from the main thread
        this.port.onmessage = (event) => {
            if (event.data.command === 'start') {
                this.isRecording = true;
            } else if (event.data.command === 'stop') {
                this.isRecording = false;
            }
        };
    }

    process(inputs, outputs, parameters) {
        // Only process if recording
        if (!this.isRecording) {
            return true;
        }

        const input = inputs[0];
        if (input.length > 0) {
            const channelData = input[0]; // Get first channel (mono)

            // Convert float32 to int16
            const int16Data = new Int16Array(channelData.length);
            for (let i = 0; i < channelData.length; i++) {
                const s = Math.max(-1, Math.min(1, channelData[i]));
                int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            // Send to main thread (ArrayBuffer is transferred, not copied)
            this.port.postMessage({
                audio: int16Data.buffer
            }, [int16Data.buffer]); // Transfer ownership for efficiency
        }

        // Keep processor alive
        return true;
    }
}

// Register the processor
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
