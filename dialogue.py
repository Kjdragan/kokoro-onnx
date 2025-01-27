"""
Multi-party dialogue demo with performance metrics
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
from typing import List, Tuple
import numpy as np

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
class GenerationMetrics:
    text_length: int
    phoneme_count: int
    generation_time: float
    audio_length: float
    realtime_factor: float

@dataclass
class PerformanceMetrics:
    warmup_time: float
    generation_metrics: List[Tuple[str, GenerationMetrics]]  # character, metrics
    total_time: float

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
    # Short warmup text for each available voice to initialize all components
    warmup_text = "Warming up the model."
    
    print("Warming up model...")
    for voice in voices[:3]:  # Use first 3 voices for warmup
        kokoro.create(warmup_text, voice=voice, speed=1.0, lang="en-us")
    
    warmup_time = time.time() - start_time
    print(f"Model warmup completed in {warmup_time:.2f}s")
    return warmup_time

def run_dialogue() -> PerformanceMetrics:
    """Run the dialogue and return performance metrics"""
    
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
    
    # Track performance metrics
    generation_metrics = []
    total_start_time = time.time()
    
    print("\nStarting dialogue...\n")
    
    # Generate and play dialogue
    for round in range(len(max(characters, key=lambda x: len(x.lines)).lines)):
        for char in characters:
            if round < len(char.lines):
                text = char.lines[round]
                print(f"\n{char.name} ({char.voice}):")
                print(f"Text ({len(text)} chars): {text}")
                
                # Generate and measure performance
                start_time = time.time()
                samples, sample_rate = kokoro.create(
                    text,
                    voice=char.voice,
                    speed=1.0,
                    lang="en-us"
                )
                generation_time = time.time() - start_time
                
                # Calculate metrics
                audio_length = len(samples) / sample_rate
                realtime_factor = audio_length / generation_time
                
                # Store metrics
                metrics = GenerationMetrics(
                    text_length=len(text),
                    phoneme_count=len(text.split()),  # Approximate
                    generation_time=generation_time,
                    audio_length=audio_length,
                    realtime_factor=realtime_factor
                )
                generation_metrics.append((char.name, metrics))
                
                # Print immediate feedback
                print(f"Generation Metrics:")
                print(f"- Text Length: {metrics.text_length} chars")
                print(f"- Generation Time: {metrics.generation_time:.2f}s")
                print(f"- Audio Length: {metrics.audio_length:.2f}s")
                print(f"- Realtime Factor: {metrics.realtime_factor:.2f}x")
                
                # Play audio
                sd.play(samples, sample_rate)
                sd.wait()
                print("-" * 80)
    
    total_time = time.time() - total_start_time
    
    return PerformanceMetrics(
        warmup_time=warmup_time,
        generation_metrics=generation_metrics,
        total_time=total_time
    )

def print_performance_summary(metrics: PerformanceMetrics):
    """Print a summary of performance metrics"""
    print("\nPerformance Summary:")
    print(f"Model Warmup Time: {metrics.warmup_time:.2f}s")
    print("\nGeneration Analysis:")
    
    # Calculate statistics per character
    char_metrics = {}
    for char, gen_metrics in metrics.generation_metrics:
        if char not in char_metrics:
            char_metrics[char] = []
        char_metrics[char].append(gen_metrics)
    
    for char, measurements in char_metrics.items():
        print(f"\n{char}:")
        text_lengths = [m.text_length for m in measurements]
        gen_times = [m.generation_time for m in measurements]
        rt_factors = [m.realtime_factor for m in measurements]
        audio_lengths = [m.audio_length for m in measurements]
        
        print(f"  Avg Text Length: {np.mean(text_lengths):.1f} chars")
        print(f"  Avg Generation Time: {np.mean(gen_times):.2f}s")
        print(f"  Avg Audio Length: {np.mean(audio_lengths):.2f}s")
        print(f"  Avg Realtime Factor: {np.mean(rt_factors):.2f}x")
        print(f"  Characters/Second: {np.mean([l/t for l,t in zip(text_lengths, gen_times)]):.1f}")
    
    # Overall statistics
    all_text_lengths = [m.text_length for _, m in metrics.generation_metrics]
    all_gen_times = [m.generation_time for _, m in metrics.generation_metrics]
    all_rt_factors = [m.realtime_factor for _, m in metrics.generation_metrics]
    
    print(f"\nOverall Statistics:")
    print(f"Total Dialogue Time: {metrics.total_time:.2f}s")
    print(f"Average Text Length: {np.mean(all_text_lengths):.1f} chars")
    print(f"Average Generation Time: {np.mean(all_gen_times):.2f}s")
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
