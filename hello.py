"""
Using kokoro-onnx with optimized GPU acceleration
"""

import os
import logging
import sounddevice as sd
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession
import onnxruntime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('kokoro_onnx')
logger.setLevel(logging.DEBUG)

def create_optimized_session():
    # Optimize session options
    sess_options = onnxruntime.SessionOptions()
    
    # Set thread count to CPU cores
    cpu_count = os.cpu_count()
    sess_options.intra_op_num_threads = cpu_count
    
    # Enable CUDA optimization
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Reduce memory copies
    sess_options.enable_cpu_mem_arena = False
    
    # Create session with CUDA provider
    providers = ['CUDAExecutionProvider']
    session = InferenceSession(
        "kokoro-v0_19.onnx",
        providers=providers,
        sess_options=sess_options
    )
    return session

try:
    # Create optimized session
    session = create_optimized_session()
    
    # Initialize Kokoro with optimized session
    kokoro = Kokoro.from_session(session, "voices.bin")
    
    # Generate audio
    print("Generating audio...")
    samples, sample_rate = kokoro.create(
        "Hello. This audio generated by kokoro using GPU acceleration!", 
        voice="af_sarah", 
        speed=1.0, 
        lang="en-us"
    )
    
    print("Playing audio...")
    sd.play(samples, sample_rate)
    sd.wait()

except Exception as e:
    print(f"Error: {e}")
    print("If you're seeing CUDA/GPU related errors, ensure your NVIDIA drivers are up to date")
