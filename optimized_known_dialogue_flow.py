"""
Multi-party dialogue demo with performance metrics and queue-based playback
Uses model warmup and different voices for each character
"""

import time
import logging
import sounddevice as sd
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import onnxruntime
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor, Future

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kokoro_onnx')
logger.setLevel(logging.DEBUG)

@dataclass
class Character:
    name: str
    voice: str
    lines: List[str]

@dataclass
class AudioChunk:
    character_name: str
    text: str
    samples: np.ndarray
    sample_rate: int
    generation_time: float
    audio_length: float
    text_length: int
    chunk_index: int
    total_chunks: int

@dataclass
class GenerationMetrics:
    text_length: int
    phoneme_count: int
    generation_time: float
    audio_length: float
    realtime_factor: float
    queue_wait_time: float  # Time spent waiting in queue
    total_latency: float    # Total time from generation start to playback start

@dataclass
class PerformanceMetrics:
    warmup_time: float
    generation_metrics: List[Tuple[str, GenerationMetrics]]
    total_time: float
    average_queue_length: float
    max_queue_length: int
    total_queue_wait_time: float

class AudioQueue:
    def __init__(self, max_size: int = 100):
        self.queue = Queue(maxsize=max_size)
        self.current_chunk: Optional[AudioChunk] = None
        self.is_playing = False
        self.total_wait_time = 0
        self.queue_lengths = []
        self.max_queue_length = 0
        self._lock = threading.Lock()

    def put(self, chunk: AudioChunk) -> float:
        """Put a chunk in the queue and return the wait time"""
        start_time = time.time()
        self.queue.put(chunk)
        wait_time = time.time() - start_time
        
        with self._lock:
            queue_length = self.queue.qsize()
            self.queue_lengths.append(queue_length)
            self.max_queue_length = max(self.max_queue_length, queue_length)
            self.total_wait_time += wait_time
        
        return wait_time

    def get(self) -> Optional[AudioChunk]:
        """Get the next chunk from the queue"""
        if not self.queue.empty():
            self.current_chunk = self.queue.get()
            return self.current_chunk
        return None

    def get_stats(self) -> Tuple[float, int, float]:
        """Return average queue length, max length, and total wait time"""
        with self._lock:
            avg_length = np.mean(self.queue_lengths) if self.queue_lengths else 0
            return avg_length, self.max_queue_length, self.total_wait_time

def create_optimized_session() -> InferenceSession:
    """Create an optimized ONNX session for GPU inference"""
    sess_options = SessionOptions()
    
    # Optimize for GPU performance
    sess_options.intra_op_num_threads = os.cpu_count()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_cpu_mem_arena = False
    
    providers = ['CUDAExecutionProvider']
    return InferenceSession(
        "kokoro-v0_19.onnx",
        providers=providers,
        sess_options=sess_options
    )

def warmup_model(kokoro: Kokoro, voices: List[str]) -> float:
    """Warm up the model and return the time taken"""
    start_time = time.time()
    warmup_text = "Warming up the model."
    
    print("Warming up model...")
    for voice in voices[:3]:
        kokoro.create(warmup_text, voice=voice, speed=1.0, lang="en-us")
    
    warmup_time = time.time() - start_time
    print(f"Model warmup completed in {warmup_time:.2f}s")
    return warmup_time

def generate_audio_chunk(
    kokoro: Kokoro,
    text: str,
    voice: str,
    character_name: str,
    chunk_index: int,
    total_chunks: int
) -> Tuple[AudioChunk, GenerationMetrics]:
    """Generate a single audio chunk with metrics"""
    start_time = time.time()
    
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=1.0,
        lang="en-us"
    )
    
    generation_time = time.time() - start_time
    audio_length = len(samples) / sample_rate
    
    chunk = AudioChunk(
        character_name=character_name,
        text=text,
        samples=samples,
        sample_rate=sample_rate,
        generation_time=generation_time,
        audio_length=audio_length,
        text_length=len(text),
        chunk_index=chunk_index,
        total_chunks=total_chunks
    )
    
    metrics = GenerationMetrics(
        text_length=len(text),
        phoneme_count=len(text.split()),  # Approximate
        generation_time=generation_time,
        audio_length=audio_length,
        realtime_factor=audio_length / generation_time,
        queue_wait_time=0,  # Will be updated when queued
        total_latency=0     # Will be updated when played
    )
    
    return chunk, metrics

def audio_player(audio_queue: AudioQueue, stop_event: threading.Event):
    """Continuously play audio from the queue"""
    while not stop_event.is_set() or not audio_queue.queue.empty():
        chunk = audio_queue.get()
        if chunk:
            print(f"\nPlaying {chunk.character_name} (Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}):")
            print(f"Text: {chunk.text}")
            print(f"Generation Time: {chunk.generation_time:.2f}s")
            print(f"Audio Length: {chunk.audio_length:.2f}s")
            print("-" * 80)
            
            sd.play(chunk.samples, chunk.sample_rate)
            sd.wait()
        else:
            time.sleep(0.1)  # Small delay when queue is empty

def run_dialogue() -> PerformanceMetrics:
    """Run the dialogue with queue-based generation and playback"""
    
    # Create session and model
    session = create_optimized_session()
    kokoro = Kokoro.from_session(session, "voices.bin")
    
    # Get available voices
    available_voices = kokoro.get_voices()
    print(f"Available voices: {available_voices}")
    
    if len(available_voices) < 3:
        raise ValueError(f"Need at least 3 voices, but only found {len(available_voices)}")
    
    # Initialize characters with available voices and longer dialogue
    characters = [
        Character("Speaker One", available_voices[0], [
            "Welcome everyone to our annual strategic planning meeting. Today, we'll be discussing our company's performance over the past fiscal year and outlining our objectives for the upcoming quarters.",
            "That's a fascinating analysis of the market trends, Speaker Three. The correlation between customer engagement metrics and our new product launches shows promising potential for expansion.",
            "Based on the comprehensive data presented today, I believe we've identified several key opportunities for growth. Let's schedule follow-up meetings with each department to develop detailed implementation plans.",
            "Before we conclude, I'd like to emphasize the exceptional work done by all teams. Your dedication and innovative approaches have significantly contributed to our success this year."
        ]),
        Character("Speaker Two", available_voices[1], [
            "Thank you for the introduction. I've prepared an extensive analysis of our quarterly performance, including detailed breakdowns of revenue streams, customer acquisition costs, and market penetration metrics.",
            "The financial projections look particularly promising, especially considering the challenging market conditions we've navigated. Our risk mitigation strategies have proven highly effective.",
            "I've also conducted a thorough competitive analysis, and our positioning in emerging markets shows remarkable potential for the next fiscal year."
        ]),
        Character("Speaker Three", available_voices[2], [
            "Our market research indicates a significant shift in consumer behavior patterns. We've observed a thirty-five percent increase in digital engagement, with particularly strong growth in mobile platform interactions.",
            "Furthermore, our customer satisfaction metrics have shown consistent improvement across all service categories, with our Net Promoter Score reaching an all-time high of seventy-eight point five.",
            "To conclude my portion of the presentation, I'd like to share our recommendations for strategic initiatives that could help us capitalize on these emerging opportunities."
        ])
    ]
    
    # Warm up the model
    warmup_time = warmup_model(kokoro, available_voices)
    
    # Initialize audio queue and metrics
    audio_queue = AudioQueue()
    generation_metrics = []
    total_start_time = time.time()
    
    # Create and start the audio player thread
    stop_event = threading.Event()
    player_thread = threading.Thread(
        target=audio_player,
        args=(audio_queue, stop_event)
    )
    player_thread.start()
    
    # Generate audio chunks concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures: List[Future] = []
        
        # Submit all generation tasks
        for round_idx in range(len(max(characters, key=lambda x: len(x.lines)).lines)):
            for char in characters:
                if round_idx < len(char.lines):
                    future = executor.submit(
                        generate_audio_chunk,
                        kokoro,
                        char.lines[round_idx],
                        char.voice,
                        char.name,
                        round_idx,
                        len(char.lines)
                    )
                    futures.append(future)
        
        # Process completed chunks in order
        for future in futures:
            chunk, metrics = future.result()
            queue_wait_time = audio_queue.put(chunk)
            
            # Update metrics with queue information
            metrics.queue_wait_time = queue_wait_time
            metrics.total_latency = metrics.generation_time + queue_wait_time
            generation_metrics.append((chunk.character_name, metrics))
    
    # Wait for all audio to finish playing
    stop_event.set()
    player_thread.join()
    
    # Calculate final metrics
    total_time = time.time() - total_start_time
    avg_queue_length, max_queue_length, total_queue_wait = audio_queue.get_stats()
    
    return PerformanceMetrics(
        warmup_time=warmup_time,
        generation_metrics=generation_metrics,
        total_time=total_time,
        average_queue_length=avg_queue_length,
        max_queue_length=max_queue_length,
        total_queue_wait_time=total_queue_wait
    )

def print_performance_summary(metrics: PerformanceMetrics):
    """Print a summary of performance metrics"""
    print("\nPerformance Summary:")
    print(f"Model Warmup Time: {metrics.warmup_time:.2f}s")
    print("\nQueue Statistics:")
    print(f"Average Queue Length: {metrics.average_queue_length:.2f}")
    print(f"Maximum Queue Length: {metrics.max_queue_length}")
    print(f"Total Queue Wait Time: {metrics.total_queue_wait_time:.2f}s")
    
    print("\nGeneration Analysis:")
    
    # Calculate statistics per character
    char_metrics: Dict[str, List[GenerationMetrics]] = {}
    for char, gen_metrics in metrics.generation_metrics:
        if char not in char_metrics:
            char_metrics[char] = []
        char_metrics[char].append(gen_metrics)
    
    for char, measurements in char_metrics.items():
        print(f"\n{char}:")
        text_lengths = [m.text_length for m in measurements]
        gen_times = [m.generation_time for m in measurements]
        queue_times = [m.queue_wait_time for m in measurements]
        total_latencies = [m.total_latency for m in measurements]
        rt_factors = [m.realtime_factor for m in measurements]
        audio_lengths = [m.audio_length for m in measurements]
        
        print(f"  Avg Text Length: {np.mean(text_lengths):.1f} chars")
        print(f"  Avg Generation Time: {np.mean(gen_times):.2f}s")
        print(f"  Avg Queue Wait Time: {np.mean(queue_times):.2f}s")
        print(f"  Avg Total Latency: {np.mean(total_latencies):.2f}s")
        print(f"  Avg Audio Length: {np.mean(audio_lengths):.2f}s")
        print(f"  Avg Realtime Factor: {np.mean(rt_factors):.2f}x")
        print(f"  Characters/Second: {np.mean([l/t for l,t in zip(text_lengths, gen_times)]):.1f}")
    
    # Overall statistics
    all_text_lengths = [m.text_length for _, m in metrics.generation_metrics]
    all_gen_times = [m.generation_time for _, m in metrics.generation_metrics]
    all_queue_times = [m.queue_wait_time for _, m in metrics.generation_metrics]
    all_latencies = [m.total_latency for _, m in metrics.generation_metrics]
    all_rt_factors = [m.realtime_factor for _, m in metrics.generation_metrics]
    
    print(f"\nOverall Statistics:")
    print(f"Total Dialogue Time: {metrics.total_time:.2f}s")
    print(f"Average Text Length: {np.mean(all_text_lengths):.1f} chars")
    print(f"Average Generation Time: {np.mean(all_gen_times):.2f}s")
    print(f"Average Queue Wait Time: {np.mean(all_queue_times):.2f}s")
    print(f"Average Total Latency: {np.mean(all_latencies):.2f}s")
    print(f"Average Realtime Factor: {np.mean(all_rt_factors):.2f}x")
    print(f"Average Characters/Second: {np.mean([l/t for l,t in zip(all_text_lengths, all_gen_times)]):.1f}")
    
    # Correlation analysis
    correlation = np.corrcoef(all_text_lengths, all_gen_times)[0,1]
    print(f"\nCorrelation between text length and generation time: {correlation:.3f}")

if __name__ == "__main__":
    try:
        metrics = run_dialogue()
        print_performance_summary(metrics)
    except Exception as e:
        print(f"Error: {e}")
        print("If you're seeing CUDA/GPU related errors, ensure your NVIDIA drivers are up to date")
