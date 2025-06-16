#!/usr/bin/env python3
"""
SuperGPU - GPU Saturation Script for Language Identification
Continuously runs language identification inference to maximize GPU utilization.
"""

import queue
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import gpu
import settings
from utils import suppress_output

class SuperGPU:
    def __init__(self):
        self.running = True
        self.model = None
        self.processor = None
        self.batch_queue = queue.Queue(maxsize=10)
        self.stats = {
            'batches_processed': 0,
            'total_samples': 0,
            'start_time': time.time()
        }
        
        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\nReceived shutdown signal. Stopping SuperGPU...")
        self.running = False
    
    def load_model(self):
        """Load the language identification model"""
        print("Loading Whisper-tiny model for language identification...")
        start_time = time.time()
        
        with suppress_output(should_suppress=True):
            model_name = settings.lang_id_model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        
        # Move to GPU and optimize
        self.model = gpu.move_to_gpu_maybe(self.model)
        self.model.eval()
        
        # Enable optimizations
        if gpu.we_have_a_gpu():
            # Try FP16 optimization - may not work with all Whisper models
            try:
                self.model = self.model.half()  # Use FP16 for more throughput
                print("✓ FP16 optimization enabled")
            except Exception as e:
                print(f"⚠️ FP16 optimization failed, staying in FP32: {e}")
            
            torch.backends.cudnn.benchmark = True
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        return self.model, self.processor
    
    def generate_synthetic_audio_batch(self, batch_size=32, duration_seconds=30):
        """Generate synthetic audio data to avoid I/O bottlenecks"""
        sample_rate = settings.sample_rate
        samples_per_audio = int(duration_seconds * sample_rate)
        
        # Generate realistic-looking audio (white noise with some structure)
        audio_batch = []
        for _ in range(batch_size):
            # Create structured noise that resembles speech patterns
            base_freq = np.random.uniform(80, 300)  # Human speech fundamental frequency range
            t = np.linspace(0, duration_seconds, samples_per_audio)
            
            # Create a more speech-like signal
            signal = (
                0.3 * np.sin(2 * np.pi * base_freq * t) +
                0.2 * np.sin(2 * np.pi * base_freq * 2 * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 3 * t) +
                0.1 * np.random.randn(samples_per_audio)
            )
            
            # Add some amplitude modulation to make it more speech-like
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
            signal *= envelope
            
            # Normalize
            signal = signal.astype(np.float32)
            signal = signal / (np.max(np.abs(signal)) + 1e-8)
            
            audio_batch.append(signal)
        
        return audio_batch
    
    def calculate_optimal_batch_size(self):
        """Calculate the largest batch size that fits in GPU memory"""
        if not gpu.we_have_a_gpu():
            return 8
        
        print("Calculating optimal batch size...")
        
        # Start with a small batch and increase until we hit memory limits
        test_batch_sizes = [8, 16, 32, 48, 64, 80, 96, 112, 128]
        optimal_batch_size = 8
        
        for batch_size in test_batch_sizes:
            try:
                print(f"Testing batch size: {batch_size}")
                
                # Generate test batch
                audio_batch = self.generate_synthetic_audio_batch(batch_size, duration_seconds=30)
                
                # Try processing
                with torch.no_grad():
                    inputs = self.processor(
                        audio_batch,
                        sampling_rate=settings.sample_rate,
                        return_tensors="pt",
                        padding="max_length"
                    )
                    inputs = gpu.move_to_gpu_maybe(inputs)
                    input_features = gpu.move_to_gpu_maybe(inputs.input_features)
                    # Note: whisper_start_transcription_token_id was from ai.py, using common Whisper value
                    decoder_input_ids = torch.tensor([[50258]] * batch_size)  # Common Whisper start token
                    decoder_input_ids = gpu.move_to_gpu_maybe(decoder_input_ids)
                    
                    # Test inference
                    model_output = self.model(input_features, decoder_input_ids=decoder_input_ids)
                    logits = model_output.logits
                    
                    optimal_batch_size = batch_size
                    print(f"✓ Batch size {batch_size} successful")
                    
                    # Clear memory
                    del inputs, input_features, decoder_input_ids, model_output, logits
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"✗ Batch size {batch_size} failed: OOM")
                    break
                else:
                    print(f"✗ Batch size {batch_size} failed: {e}")
                    break
            except Exception as e:
                print(f"✗ Batch size {batch_size} failed: {e}")
                break
        
        print(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def identify_language_batch_intensive(self, audio_batch):
        """Intensive language identification with maximum GPU utilization"""
        batch_size = len(audio_batch)
        
        with torch.no_grad():
            # Process audio
            inputs = self.processor(
                audio_batch,
                sampling_rate=settings.sample_rate,
                return_tensors="pt",
                padding="max_length"
            )
            
            inputs = gpu.move_to_gpu_maybe(inputs)
            input_features = gpu.move_to_gpu_maybe(inputs.input_features)
            
            # Note: whisper_start_transcription_token_id was from ai.py, using common Whisper value  
            decoder_input_ids = torch.tensor([[50258]] * batch_size)  # Common Whisper start token
            decoder_input_ids = gpu.move_to_gpu_maybe(decoder_input_ids)
            
            # Run inference
            model_output = self.model(input_features, decoder_input_ids=decoder_input_ids)
            logits = model_output.logits[:, 0, :]
            
            # Process results (simulate full pipeline)
            # Note: ai.whisper_token_ids and ai.whisper_tokens were from ai.py module
            # Commenting out language detection logic since ai.py is removed
            languages = []
            for i in range(batch_size):
                # lang_logits = logits[i, ai.whisper_token_ids]
                # lang_probs = torch.softmax(lang_logits, dim=-1)
                # max_prob_idx = lang_probs.argmax()
                # detected_lang = ai.whisper_tokens[max_prob_idx]
                languages.append("unknown")  # Placeholder since ai.py constants not available
            
            return languages
    
    def batch_producer(self, batch_size, max_batches_in_queue=5):
        """Continuously generate batches of synthetic audio"""
        print(f"Starting batch producer with batch size: {batch_size}")
        
        while self.running:
            try:
                if self.batch_queue.qsize() < max_batches_in_queue:
                    audio_batch = self.generate_synthetic_audio_batch(batch_size)
                    self.batch_queue.put(audio_batch, timeout=1.0)
                else:
                    time.sleep(0.01)  # Brief pause if queue is full
            except queue.Full:
                continue
            except Exception as e:
                print(f"Batch producer error: {e}")
                break
    
    def gpu_consumer(self, num_consumers=2):
        """Consume batches and run GPU inference"""
        print(f"Starting {num_consumers} GPU consumers")
        
        def consumer_worker():
            while self.running:
                try:
                    audio_batch = self.batch_queue.get(timeout=1.0)
                    
                    # Run inference
                    languages = self.identify_language_batch_intensive(audio_batch)
                    
                    # Update stats
                    self.stats['batches_processed'] += 1
                    self.stats['total_samples'] += len(audio_batch)
                    
                    self.batch_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"GPU consumer error: {e}")
                    continue
        
        # Start consumer threads
        consumer_threads = []
        for i in range(num_consumers):
            thread = threading.Thread(target=consumer_worker, daemon=True)
            thread.start()
            consumer_threads.append(thread)
        
        return consumer_threads
    
    def print_stats(self):
        """Print performance statistics"""
        elapsed = time.time() - self.stats['start_time']
        batches_per_sec = self.stats['batches_processed'] / elapsed if elapsed > 0 else 0
        samples_per_sec = self.stats['total_samples'] / elapsed if elapsed > 0 else 0
        
        print(f"\n--- SuperGPU Stats ---")
        print(f"Elapsed time: {elapsed:.1f}s")
        print(f"Batches processed: {self.stats['batches_processed']}")
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Batches/sec: {batches_per_sec:.2f}")
        print(f"Samples/sec: {samples_per_sec:.1f}")
        
        if gpu.we_have_a_gpu():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def run(self):
        """Main execution loop"""
        print("=" * 60)
        print("SuperGPU - GPU Saturation Script")
        print("=" * 60)
        
        if not gpu.we_have_a_gpu():
            print("ERROR: No GPU detected. This script requires a CUDA-capable GPU.")
            return
        
        # Load model
        self.load_model()
        
        # Calculate optimal batch size
        batch_size = self.calculate_optimal_batch_size()
        
        # Start producer thread
        producer_thread = threading.Thread(
            target=self.batch_producer, 
            args=(batch_size,), 
            daemon=True
        )
        producer_thread.start()
        
        # Start consumer threads (multiple for maximum GPU utilization)
        consumer_threads = self.gpu_consumer(num_consumers=2)
        
        print(f"\nSuperGPU running with batch size {batch_size}")
        print("Press Ctrl+C to stop\n")
        
        # Stats reporting loop
        last_stats_time = time.time()
        while self.running:
            try:
                time.sleep(5)  # Print stats every 5 seconds
                
                current_time = time.time()
                if current_time - last_stats_time >= 5:
                    self.print_stats()
                    gpu.print_gpu_memory_usage()
                    last_stats_time = current_time
                    
            except KeyboardInterrupt:
                break
        
        print("\nShutting down SuperGPU...")
        self.running = False
        
        # Wait for threads to finish
        producer_thread.join(timeout=2)
        for thread in consumer_threads:
            thread.join(timeout=2)
        
        # Final stats
        self.print_stats()
        print("SuperGPU stopped.")

def main():
    supergpu = SuperGPU()
    supergpu.run()

if __name__ == "__main__":
    main() 