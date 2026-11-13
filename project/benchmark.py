#!/usr/bin/env python3
"""
SENTIX Comprehensive AI & Inference Engine Benchmark
Measures latency (Mean, P50, P95, P99), throughput (FPS/ops), and memory footprint
across ASL Alphabet (Model A), ASL Word (Model B), Text Affect/Emotion Models,
and Real-Time Landmark Smoothing Pipelines.
"""

import os
import sys
import time
import tracemalloc
import numpy as np
import torch
from pathlib import Path

# Add project directory to path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.asl.config import (
    ALPHABET_BEST_MODEL,
    WORD_BEST_MODEL,
    MLP_INPUT_DIM,
    MLP_NUM_CLASSES,
    TRANSFORMER_INPUT_DIM,
    TRANSFORMER_SEQ_LEN
)
from backend.asl.alphabet_model import FingerspellingMLP
from backend.asl.word_model import PoseTransformer, SpatioTemporalAttentionModel
from backend.asl.session_manager import ASLSession
from backend.utils import normalize_text_affect, compute_valence_arousal


def format_stats(latencies_ms):
    """Compute and format latency statistics."""
    arr = np.array(latencies_ms)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "fps": float(1000.0 / np.mean(arr)) if np.mean(arr) > 0 else 0.0,
    }


def benchmark_alphabet_model(device="cpu", iterations=150):
    """Benchmark Model A (Alphabet 4-Layer MLP, 126 dims -> 29 classes)."""
    model = FingerspellingMLP(input_dim=MLP_INPUT_DIM, num_classes=MLP_NUM_CLASSES).to(device)

    if ALPHABET_BEST_MODEL.exists():
        state = torch.load(ALPHABET_BEST_MODEL, map_location=device, weights_only=False)
        model = FingerspellingMLP(state.get("input_dim", MLP_INPUT_DIM), state.get("num_classes", MLP_NUM_CLASSES)).to(device)
        model.load_state_dict(state.get("model_state_dict", state))

    model.eval()
    dummy_input = torch.randn(1, MLP_INPUT_DIM, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return format_stats(latencies)


def benchmark_word_model(device="cpu", iterations=100):
    """Benchmark Model B (SpatioTemporalAttentionModel / PoseTransformer)."""
    if WORD_BEST_MODEL.exists():
        try:
            ckpt = torch.load(WORD_BEST_MODEL, map_location=device, weights_only=False)
            model_type = ckpt.get("model_type", "SpatioTemporalAttentionModel")
            num_classes = ckpt.get("num_classes", 54)
            if model_type == "SpatioTemporalAttentionModel":
                model = SpatioTemporalAttentionModel(
                    input_dim=ckpt.get("input_dim", 225),
                    seq_len=ckpt.get("seq_len", TRANSFORMER_SEQ_LEN),
                    d_model=ckpt.get("d_model", 256),
                    hidden_dim=ckpt.get("hidden_dim", 128),
                    num_classes=num_classes,
                    n_heads=ckpt.get("n_heads", 4),
                )
            else:
                model = PoseTransformer(
                    input_dim=ckpt.get("input_dim", 225),
                    seq_len=ckpt.get("seq_len", TRANSFORMER_SEQ_LEN),
                    d_model=ckpt.get("d_model", 512),
                    n_heads=ckpt.get("n_heads", 8),
                    n_layers=ckpt.get("n_layers", 6),
                    d_ff=ckpt.get("d_ff", 2048),
                    num_classes=num_classes,
                )
            model.load_state_dict(ckpt["model_state_dict"])
        except Exception:
            model = SpatioTemporalAttentionModel(
                input_dim=TRANSFORMER_INPUT_DIM,
                seq_len=TRANSFORMER_SEQ_LEN,
                d_model=256,
                hidden_dim=128,
                num_classes=54,
                n_heads=4,
            )
    else:
        model = SpatioTemporalAttentionModel(
            input_dim=TRANSFORMER_INPUT_DIM,
            seq_len=TRANSFORMER_SEQ_LEN,
            d_model=256,
            hidden_dim=128,
            num_classes=54,
            n_heads=4,
        )

    model.to(device)
    model.eval()

    dummy_seq = torch.randn(1, TRANSFORMER_SEQ_LEN, TRANSFORMER_INPUT_DIM, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_seq)

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy_seq)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    return format_stats(latencies)


def benchmark_landmark_smoothing(iterations=300):
    """Benchmark real-time adaptive EMA landmark smoothing & entropy computation."""
    session = ASLSession("bench_session")
    dummy_hand = np.random.randn(126).astype(np.float32)
    dummy_full = np.random.randn(225).astype(np.float32)
    dummy_probs = np.random.dirichlet(np.ones(54)).astype(np.float32)

    # Hand smoothing
    hand_latencies = []
    for _ in range(iterations):
        curr_hand = dummy_hand + np.random.normal(0, 0.01, 126).astype(np.float32)
        t0 = time.perf_counter()
        _ = session.smooth_hand_landmarks(curr_hand)
        t1 = time.perf_counter()
        hand_latencies.append((t1 - t0) * 1000.0)

    # Full smoothing
    full_latencies = []
    for _ in range(iterations):
        curr_full = dummy_full + np.random.normal(0, 0.01, 225).astype(np.float32)
        t0 = time.perf_counter()
        _ = session.smooth_full_landmarks(curr_full)
        t1 = time.perf_counter()
        full_latencies.append((t1 - t0) * 1000.0)

    # Entropy computation
    entropy_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = ASLSession.compute_normalized_entropy(dummy_probs)
        t1 = time.perf_counter()
        entropy_latencies.append((t1 - t0) * 1000.0)

    return {
        "hand_smoothing": format_stats(hand_latencies),
        "full_smoothing": format_stats(full_latencies),
        "entropy_gating": format_stats(entropy_latencies),
    }


def benchmark_affect_pipeline(iterations=200):
    """Benchmark text affect normalization & Russell's Circumplex coordinate mapping."""
    sample_text = "OMG I am so proud of this project!! ❤️🔥🚀 thank you sm"
    sample_scores = {"joy": 0.85, "neutral": 0.10, "surprise": 0.05}

    text_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = normalize_text_affect(sample_text)
        t1 = time.perf_counter()
        text_latencies.append((t1 - t0) * 1000.0)

    circumplex_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = compute_valence_arousal(sample_scores)
        t1 = time.perf_counter()
        circumplex_latencies.append((t1 - t0) * 1000.0)

    return {
        "text_normalization": format_stats(text_latencies),
        "circumplex_mapping": format_stats(circumplex_latencies),
    }


def run_all_benchmarks():
    """Run comprehensive benchmark suite and output structured report."""
    print("=" * 72)
    print("      SENTIX ADVANCED INFERENCE & PIPELINE PERFORMANCE BENCHMARK       ")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device Target:     {device.upper()}")
    if device == "cuda":
        print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
    print(f"Python Runtime:    {sys.version.split()[0]}")
    print(f"PyTorch Version:   {torch.__version__}")
    print("-" * 72)

    tracemalloc.start()

    print("[1/4] Benchmarking ASL Model A (Alphabet 4-Layer MLP)...")
    alpha_stats = benchmark_alphabet_model(device=device)

    print("[2/4] Benchmarking ASL Model B (Word-Level Model)...")
    word_stats = benchmark_word_model(device=device)

    print("[3/4] Benchmarking Landmark Smoothing & Entropy Gating...")
    smooth_stats = benchmark_landmark_smoothing()

    print("[4/4] Benchmarking Text Affect & Circumplex Projection...")
    affect_stats = benchmark_affect_pipeline()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "=" * 72)
    print(f"{'Component / Subsystem':<35} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Throughput':<12}")
    print("-" * 72)

    print(f"{'Model A: Alphabet MLP (126-dim)':<35} | {alpha_stats['mean']:<10.3f} | {alpha_stats['p95']:<10.3f} | {alpha_stats['fps']:<10.1f} FPS")
    print(f"{'Model B: Word Model (32x225)':<35} | {word_stats['mean']:<10.3f} | {word_stats['p95']:<10.3f} | {word_stats['fps']:<10.1f} FPS")
    print(f"{'Landmark Smoothing (Hand)':<35} | {smooth_stats['hand_smoothing']['mean']:<10.4f} | {smooth_stats['hand_smoothing']['p95']:<10.4f} | {smooth_stats['hand_smoothing']['fps']:<10.0f} ops/s")
    print(f"{'Landmark Smoothing (Full Body)':<35} | {smooth_stats['full_smoothing']['mean']:<10.4f} | {smooth_stats['full_smoothing']['p95']:<10.4f} | {smooth_stats['full_smoothing']['fps']:<10.0f} ops/s")
    print(f"{'Shannon Entropy OOD Gating':<35} | {smooth_stats['entropy_gating']['mean']:<10.4f} | {smooth_stats['entropy_gating']['p95']:<10.4f} | {smooth_stats['entropy_gating']['fps']:<10.0f} ops/s")
    print(f"{'Text Slang/Emoji Normalization':<35} | {affect_stats['text_normalization']['mean']:<10.4f} | {affect_stats['text_normalization']['p95']:<10.4f} | {affect_stats['text_normalization']['fps']:<10.0f} ops/s")
    print(f"{'Valence-Arousal Circumplex':<35} | {affect_stats['circumplex_mapping']['mean']:<10.4f} | {affect_stats['circumplex_mapping']['p95']:<10.4f} | {affect_stats['circumplex_mapping']['fps']:<10.0f} ops/s")

    print("-" * 72)
    print(f"Memory Allocation Peak: {peak_mem / (1024 * 1024):.2f} MB")
    print("=" * 72)

    return {
        "alpha": alpha_stats,
        "word": word_stats,
        "smoothing": smooth_stats,
        "affect": affect_stats,
        "peak_mem_mb": peak_mem / (1024 * 1024)
    }


if __name__ == "__main__":
    run_all_benchmarks()
