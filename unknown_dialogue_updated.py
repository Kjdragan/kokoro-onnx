from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time
import queue
import threading
import sounddevice as sd
import numpy as np
import os
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel


@dataclass
class AudioChunk:
    character_name: str
    text: str
    samples: np.ndarray
    generation_time: float
    queue_wait_time: float = 0.0
    audio_length: float = 0.0
    total_latency: float = 0.0


@dataclass
class PerformanceMetrics:
    total_dialogue_time: float = 0.0
    avg_text_length: float = 0.0
    avg_generation_time: float = 0.0
    avg_queue_wait_time: float = 0.0
    avg_total_latency: float = 0.0
    avg_realtime_factor: float = 0.0
    avg_chars_per_second: float = 0.0
    correlation_text_gen_time: float = 0.0
    max_queue_length: int = 0
    avg_queue_length: float = 0.0
    per_character_metrics: Dict[str, Dict] = field(default_factory=dict)


def create_optimized_session() -> InferenceSession:
    """Create an optimized ONNX session for GPU inference"""
    sess_options = SessionOptions()

    # Optimize for GPU performance
    sess_options.intra_op_num_threads = os.cpu_count()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_cpu_mem_arena = False

    providers = ["CUDAExecutionProvider"]
    return InferenceSession(
        "kokoro-v0_19.onnx", providers=providers, sess_options=sess_options
    )


class DynamicDialogueSystem:
    def __init__(self, sample_rate: int = 24000, speed: float = 1.0):
        session = create_optimized_session()
        self.tts = Kokoro.from_session(session, "voices.bin")
        self.sample_rate = sample_rate
        self.speed = speed
        self.audio_queue = queue.Queue()
        self.current_speaker: Optional[str] = None
        self.speaker_lock = threading.Lock()
        self.is_playing = False
        self.player_thread = None
        self.metrics = PerformanceMetrics()
        self.queue_lengths = []
        self.generation_times = []
        self.text_lengths = []
        self.character_metrics: Dict[str, List[Dict]] = {}
        self.currently_playing = threading.Event()
        self.last_speaker_end_time = None  # Track when last speaker finished

    def generate_audio_chunk(
        self, character_name: str, text: str, voice: str
    ) -> AudioChunk:
        start_time = time.time()
        samples, sample_rate = self.tts.create(
            text, voice=voice, speed=self.speed, lang="en-us"
        )
        generation_time = time.time() - start_time

        chunk = AudioChunk(
            character_name=character_name,
            text=text,
            samples=samples,
            generation_time=generation_time,
            audio_length=len(samples) / sample_rate,
        )

        self.generation_times.append(generation_time)
        self.text_lengths.append(len(text))

        if character_name not in self.character_metrics:
            self.character_metrics[character_name] = []

        self.character_metrics[character_name].append(
            {
                "text_length": len(text),
                "generation_time": generation_time,
                "audio_length": chunk.audio_length,
            }
        )

        return chunk

    def update_metrics(self, chunk: AudioChunk):
        self.queue_lengths.append(self.audio_queue.qsize())

        if len(self.queue_lengths) > 0:
            self.metrics.avg_queue_length = sum(self.queue_lengths) / len(
                self.queue_lengths
            )
            self.metrics.max_queue_length = max(self.queue_lengths)

        if len(self.text_lengths) > 0 and len(self.generation_times) > 0:
            self.metrics.avg_text_length = sum(self.text_lengths) / len(
                self.text_lengths
            )
            self.metrics.avg_generation_time = sum(self.generation_times) / len(
                self.generation_times
            )

            # Calculate correlation between text length and generation time
            if len(self.text_lengths) > 1:
                correlation = np.corrcoef(self.text_lengths, self.generation_times)[
                    0, 1
                ]
                self.metrics.correlation_text_gen_time = correlation

        # Update per-character metrics
        for char_name, char_data in self.character_metrics.items():
            if not char_data:
                continue

            avg_gen_time = sum(d["generation_time"] for d in char_data) / len(char_data)
            avg_text_len = sum(d["text_length"] for d in char_data) / len(char_data)
            avg_audio_len = sum(d["audio_length"] for d in char_data) / len(char_data)

            self.metrics.per_character_metrics[char_name] = {
                "avg_text_length": avg_text_len,
                "avg_generation_time": avg_gen_time,
                "avg_audio_length": avg_audio_len,
                "avg_realtime_factor": avg_audio_len / avg_gen_time,
                "chars_per_second": avg_text_len / avg_gen_time,
            }

    def play_audio(self):
        self.is_playing = True
        while self.is_playing:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                print(f"\nPlaying {chunk.character_name}:")
                print(f"Text: {chunk.text}")
                print(f"Generation Time: {chunk.generation_time:.2f}s")
                print(f"Audio Length: {chunk.audio_length:.2f}s")
                
                if self.last_speaker_end_time is not None:
                    latency = time.time() - self.last_speaker_end_time
                    print(f"Latency since last speaker: {latency:.2f}s")
                print("-" * 80)

                # Mark that we're currently playing audio
                self.currently_playing.set()
                
                # Play the audio
                sd.play(chunk.samples, self.sample_rate)
                sd.wait()
                
                # Update last speaker end time
                self.last_speaker_end_time = time.time()
                
                # Clear the playing flag and allow next speaker
                self.currently_playing.clear()
                with self.speaker_lock:
                    self.current_speaker = None

                self.update_metrics(chunk)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio playback: {e}")
                break

    def can_speak(self, character_name: str) -> bool:
        """Check if a character can start speaking based on queue state"""
        with self.speaker_lock:
            # Only allow speaking if no one is currently speaking or processing
            if self.currently_playing.is_set() or not self.audio_queue.empty():
                return False
                
            self.current_speaker = character_name
            return True

    def add_dialogue(self, character_name: str, text: str, voice: str) -> bool:
        """
        Attempt to add dialogue for a character. Returns True if successful, False if the character
        must wait their turn.
        """
        if not self.can_speak(character_name):
            return False

        # Only generate audio when it's actually time to speak
        chunk = self.generate_audio_chunk(character_name, text, voice)
        self.audio_queue.put(chunk)

        return True

    def start_playback(self):
        if self.player_thread is None or not self.player_thread.is_alive():
            self.player_thread = threading.Thread(target=self.play_audio)
            self.player_thread.start()

    def stop_playback(self):
        self.is_playing = False
        if self.player_thread:
            self.player_thread.join()

    def stop(self):
        self.stop_playback()


def print_performance_summary(metrics: PerformanceMetrics):
    print("\nPerformance Summary:")
    print(f"\nQueue Statistics:")
    print(f"Average Queue Length: {metrics.avg_queue_length:.2f}")
    print(f"Maximum Queue Length: {metrics.max_queue_length}")

    print("\nGeneration Analysis:")
    for char_name, char_metrics in metrics.per_character_metrics.items():
        print(f"\n{char_name}:")
        print(f"  Avg Text Length: {char_metrics['avg_text_length']:.1f} chars")
        print(f"  Avg Generation Time: {char_metrics['avg_generation_time']:.2f}s")
        print(f"  Avg Audio Length: {char_metrics['avg_audio_length']:.2f}s")
        print(f"  Avg Realtime Factor: {char_metrics['avg_realtime_factor']:.2f}x")
        print(f"  Characters/Second: {char_metrics['chars_per_second']:.1f}")

    print("\nOverall Statistics:")
    print(f"Average Text Length: {metrics.avg_text_length:.1f} chars")
    print(f"Average Generation Time: {metrics.avg_generation_time:.2f}s")
    print(f"Average Queue Wait Time: {metrics.avg_queue_wait_time:.2f}s")
    print(
        f"Correlation between text length and generation time: {metrics.correlation_text_gen_time:.3f}"
    )


def main():
    # Initialize the dialogue system
    system = DynamicDialogueSystem()
    
    # Test dialogue with increased length and more iterations
    dialogues = [
        ("af_bella", "Hello everyone! I'm testing this real-time dialogue system with a longer message to see how it handles extended conversations and natural timing between speakers.", "af_bella"),
        ("am_adam", "I appreciate the thoroughness of your testing approach. Let me respond with an equally detailed message to ensure we're getting good data about the system's performance with varying text lengths.", "am_adam"),
        ("af_sarah", "I'll keep my response brief to add variety to our testing methodology!", "af_sarah"),
        ("af_bella", "Now I'm returning to the conversation with another substantial message to test how the system manages speaker transitions and maintains proper timing between participants.", "af_bella"),
        ("am_adam", "Testing the system's capabilities with rapid responses and examining how it handles the processing and queuing of messages in a dynamic conversation environment.", "am_adam"),
        # Additional iterations
        ("af_sarah", "Adding some more complexity to our testing scenario with this medium-length response.", "af_sarah"),
        ("af_bella", "Let's continue this conversation to gather more data about the system's performance over an extended period of dialogue exchange.", "af_bella"),
        ("am_adam", "I concur with the extended testing approach. It's important to verify the system's stability with longer conversations.", "am_adam"),
        ("af_sarah", "Introducing more variability in message length to stress test the system!", "af_sarah"),
        ("af_bella", "As we near the end of our test, let's ensure we've covered all the important aspects of real-time dialogue management.", "af_bella")
    ]

    print("\nWarming up model...")
    start_time = time.time()
    system.tts.create("Warmup text", voice="af", speed=1.0, lang="en-us")  # Warm up the model
    print(f"Model warmup completed in {time.time() - start_time:.2f}s\n")

    # Start the audio playback thread
    system.start_playback()
    
    # Record start time for total dialogue duration
    start_dialogue_time = time.time()

    try:
        for character_name, text, voice in dialogues:
            while True:
                try:
                    if system.can_speak(character_name):
                        print(f"\n{'='*40}")
                        print(f"Processing: {character_name}")
                        print(f"Message: {text}")
                        print(f"{'='*40}\n")
                        
                        # Record generation start time
                        gen_start = time.time()
                        
                        success = system.add_dialogue(character_name, text, voice)
                        if success:
                            print(f"✓ Successfully queued {character_name}")
                            break
                    else:
                        print(f"\r{character_name} waiting...", end="")
                        time.sleep(0.01)  # Small sleep to prevent CPU spinning
                except Exception as e:
                    print(f"Error processing dialogue: {e}")
                    break

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Calculate total dialogue time
        total_time = time.time() - start_dialogue_time
        system.metrics.total_dialogue_time = total_time
        
        # Stop the system
        system.stop()
        
        # Print final metrics
        print(f"\n{'='*80}")
        print("Test Completion Summary:")
        print(f"Total Runtime: {total_time:.2f}s")
        print(f"{'='*80}\n")
        print_performance_summary(system.metrics)


if __name__ == "__main__":
    main()
