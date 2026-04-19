# ruff: noqa: F821
# F821 is silenced because ruff's scope analysis incorrectly flags variables
# referenced inside lambdas when those names are `del`-ed later in the enclosing scope.
# the lambdas here are always invoked before the `del`, so the references are valid.
import argparse
import gc
import os
import tempfile
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# --- Constants for Qwen-TTS Approximation (1.7B Model) ---
HIDDEN_SIZE = 2048
NUM_HEADS = 16
HEAD_DIM = 128
INTERMEDIATE_SIZE = 5632
NUM_LAYERS = 24

# benchmark tuning
MIN_BENCH_TIME = 0.1  # minimum seconds per measurement
MAX_BATCH_DOUBLINGS = 6  # limit batch size growth


def format_bytes(size):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


def get_system_memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError):
        return 64 * 1024**3


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def sync(device_type):
    if "cuda" in device_type:
        torch.cuda.synchronize()


def timed_run(fn, device_type, min_iters=3):
    """Run fn enough times to get stable measurement."""
    sync(device_type)
    # warmup
    fn()
    sync(device_type)
    # determine iteration count
    start = time.time()
    fn()
    sync(device_type)
    single_t = time.time() - start
    iters = max(min_iters, int(MIN_BENCH_TIME / max(single_t, 1e-6)))
    # actual benchmark
    start = time.time()
    for _ in range(iters):
        fn()
    sync(device_type)
    return (time.time() - start) / iters


# --- Benchmarks ---


def benchmark_disk_io(size_mb=256):
    print(f"\n{'='*60}")
    print("Benchmark: Disk I/O")
    print(f"{'='*60}")
    size_bytes = size_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(delete=True) as f:
        # write
        data = os.urandom(size_bytes)
        start = time.time()
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        write_t = time.time() - start
        # read
        f.seek(0)
        os.posix_fadvise(f.fileno(), 0, size_bytes, os.POSIX_FADV_DONTNEED)
        start = time.time()
        _ = f.read()
        read_t = time.time() - start
        print(
            f"Write: {size_mb / write_t:.0f} MB/s | Read: {size_mb / read_t:.0f} MB/s"
        )


def benchmark_h2d_transfer(device_type, size_mb=256):
    if device_type == "cpu":
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: Host-to-Device Transfer ({device_type.upper()})")
    print(f"{'='*60}")
    cleanup()
    device = torch.device(device_type)
    x_cpu = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32)
    x_pinned = x_cpu.pin_memory()
    # pageable
    avg_t = timed_run(lambda: x_cpu.to(device), device_type)
    print(f"Pageable: {size_mb / avg_t:.0f} MB/s")
    # pinned
    avg_t = timed_run(lambda: x_pinned.to(device), device_type)
    print(f"Pinned:   {size_mb / avg_t:.0f} MB/s")
    del x_cpu, x_pinned


def benchmark_gemm(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: GEMM on {device_type.upper()} ({dtype})")
    print(f"{'Batch':<6} {'Time':>8} {'TFLOPS':>8}")
    print(f"{'-'*24}")
    cleanup()
    seq_len = 2048
    elem_size = 2 if dtype != torch.float32 else 4
    for i in range(MAX_BATCH_DOUBLINGS):
        batch_size = 1 << i
        m, k, n = batch_size * seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE
        if (m * k + k * n + m * n) * elem_size > max_mem:
            break
        try:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)
            avg_t = timed_run(lambda: torch.matmul(a, b), device_type)
            tflops = (2 * m * n * k) / avg_t / 1e12
            print(f"{batch_size:<6} {avg_t*1000:>7.1f}ms {tflops:>7.1f}")
            del a, b
        except Exception:
            break


def benchmark_attention(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: Attention on {device_type.upper()} ({dtype})")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    configs = (
        [
            ("Flash", [SDPBackend.FLASH_ATTENTION]),
            ("MemEff", [SDPBackend.EFFICIENT_ATTENTION]),
        ]
        if "cuda" in device_type
        else [("Default", None)]
    )
    cleanup()
    seq_len = 2048
    elem_size = 2 if dtype != torch.float32 else 4
    for name, bk in configs:
        print(f"--- {name} ---")
        print(f"{'Batch':<6} {'Time':>8} {'Tok/s':>10}")
        print(f"{'-'*26}")
        ctx = sdpa_kernel(bk) if bk else nullcontext()
        for i in range(MAX_BATCH_DOUBLINGS):
            batch_size = 1 << i
            if 3 * batch_size * NUM_HEADS * seq_len * HEAD_DIM * elem_size > max_mem:
                break
            try:
                q = torch.randn(
                    batch_size, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype
                )
                k = torch.randn(
                    batch_size, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype
                )
                v = torch.randn(
                    batch_size, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype
                )
                with ctx:
                    avg_t = timed_run(
                        lambda: F.scaled_dot_product_attention(q, k, v), device_type
                    )
                tok_s = (batch_size * seq_len) / avg_t
                print(f"{batch_size:<6} {avg_t*1000:>7.1f}ms {tok_s/1000:>9.0f}k")
                del q, k, v
            except Exception:
                break


def benchmark_conv1d(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: Conv1d (Vocoder) on {device_type.upper()} ({dtype})")
    print(f"{'Batch':<6} {'Time':>8}")
    print(f"{'-'*16}")
    cleanup()
    channels, length, kernel = 512, 2000, 7
    elem_size = 2 if dtype != torch.float32 else 4
    conv = torch.nn.Conv1d(channels, channels, kernel, padding=3).to(
        device, dtype=dtype
    )
    for i in range(MAX_BATCH_DOUBLINGS):
        batch_size = 1 << i
        if 2 * batch_size * channels * length * elem_size > max_mem:
            break
        try:
            x = torch.randn(batch_size, channels, length, device=device, dtype=dtype)
            avg_t = timed_run(lambda: conv(x), device_type)
            print(f"{batch_size:<6} {avg_t*1000:>7.2f}ms")
            del x
        except Exception:
            break
    del conv


def benchmark_conv_transpose1d(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: ConvTranspose1d (Upsample) on {device_type.upper()} ({dtype})")
    print(f"{'Config':<20} {'Time':>8} {'OutLen':>8}")
    print(f"{'-'*38}")
    cleanup()
    configs = [(512, 256, 16, 8), (256, 128, 16, 8), (128, 64, 4, 2), (64, 32, 4, 2)]
    length = 100
    for in_ch, out_ch, kernel, stride in configs:
        try:
            x = torch.randn(1, in_ch, length, device=device, dtype=dtype)
            conv_t = torch.nn.ConvTranspose1d(in_ch, out_ch, kernel, stride).to(
                device, dtype=dtype
            )
            out = conv_t(x)
            avg_t = timed_run(lambda: conv_t(x), device_type)
            print(
                f"{in_ch}->{out_ch} k{kernel}s{stride:<6} {avg_t*1000:>7.2f}ms {out.shape[-1]:>8}"
            )
            length = out.shape[-1]
            del x, conv_t, out
        except Exception as e:
            print(f"{in_ch}->{out_ch}: FAILED ({e})")
            break


def benchmark_layernorm(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: LayerNorm on {device_type.upper()} ({dtype})")
    print(f"{'Batch':<6} {'Time':>8} {'BW':>10}")
    print(f"{'-'*26}")
    cleanup()
    seq_len = 2048
    elem_size = 2 if dtype != torch.float32 else 4
    ln = torch.nn.LayerNorm(HIDDEN_SIZE).to(device, dtype=dtype)
    for i in range(MAX_BATCH_DOUBLINGS):
        batch_size = 1 << i
        if batch_size * seq_len * HIDDEN_SIZE * elem_size > max_mem:
            break
        try:
            x = torch.randn(
                batch_size, seq_len, HIDDEN_SIZE, device=device, dtype=dtype
            )
            avg_t = timed_run(lambda: ln(x), device_type)
            bw = 2 * x.numel() * x.element_size() / avg_t / 1e9
            print(f"{batch_size:<6} {avg_t*1000:>7.2f}ms {bw:>8.0f}GB/s")
            del x
        except Exception:
            break
    del ln


def benchmark_audio_transforms(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: STFT on {device_type.upper()} ({dtype})")
    print(f"{'Duration':<10} {'Time':>8} {'RTF':>8}")
    print(f"{'-'*28}")
    cleanup()
    sr, n_fft = 24000, 1024
    window = torch.hann_window(n_fft, device=device)
    for duration in [1, 5, 10]:
        try:
            x = torch.randn(1, sr * duration, device=device, dtype=dtype)
            avg_t = timed_run(
                lambda: torch.stft(
                    x.float(), n_fft, window=window, return_complex=True
                ),
                device_type,
            )
            rtf = duration / avg_t
            print(f"{duration}s{'':<7} {avg_t*1000:>7.1f}ms {rtf:>7.0f}x")
            del x
        except Exception:
            break


def benchmark_mel_spectrogram(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: Mel Spectrogram on {device_type.upper()} ({dtype})")
    print(f"{'Duration':<10} {'Time':>8} {'Frames':>8} {'RTF':>8}")
    print(f"{'-'*36}")
    cleanup()
    sr, n_fft, n_mels, hop = 24000, 1024, 80, 256
    # build mel filterbank
    mel_basis = torch.zeros(n_mels, n_fft // 2 + 1, device=device, dtype=torch.float32)
    mel_max = 2595 * torch.log10(torch.tensor(1 + sr / 2 / 700))
    mel_pts = torch.linspace(0, mel_max.item(), n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = (hz_pts / sr * n_fft).long()
    for i in range(n_mels):
        if bins[i + 1] > bins[i]:
            mel_basis[i, bins[i] : bins[i + 1]] = torch.linspace(
                0, 1, bins[i + 1] - bins[i]
            )
        if bins[i + 2] > bins[i + 1]:
            mel_basis[i, bins[i + 1] : bins[i + 2]] = torch.linspace(
                1, 0, bins[i + 2] - bins[i + 1]
            )
    window = torch.hann_window(n_fft, device=device)
    for duration in [1, 5, 10]:
        try:
            audio = torch.randn(1, sr * duration, device=device, dtype=torch.float32)

            def compute_mel():
                spec = torch.stft(audio, n_fft, hop, window=window, return_complex=True)
                return torch.matmul(mel_basis, spec.abs().squeeze(0))

            mel = compute_mel()
            avg_t = timed_run(compute_mel, device_type)
            print(
                f"{duration}s{'':<7} {avg_t*1000:>7.1f}ms "
                f"{mel.shape[-1]:>8} {duration/avg_t:>7.0f}x"
            )
            del audio, mel
        except Exception:
            break


def benchmark_kv_cache(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: KV-Cache on {device_type.upper()} ({dtype})")
    print(f"{'SeqLen':<8} {'Size':>10} {'Read BW':>10} {'Write BW':>10}")
    print(f"{'-'*40}")
    cleanup()
    elem_size = 2 if dtype != torch.float32 else 4
    for seq_len in [256, 1024, 4096]:
        cache_bytes = NUM_LAYERS * 2 * NUM_HEADS * seq_len * HEAD_DIM * elem_size
        if cache_bytes > max_mem:
            break
        try:
            kv = torch.randn(
                NUM_LAYERS,
                2,
                1,
                NUM_HEADS,
                seq_len,
                HEAD_DIM,
                device=device,
                dtype=dtype,
            )
            new_kv = torch.randn(
                NUM_LAYERS, 2, 1, NUM_HEADS, 1, HEAD_DIM, device=device, dtype=dtype
            )
            # read bandwidth (clone full cache)
            read_t = timed_run(lambda: kv.clone(), device_type)
            read_bw = cache_bytes / read_t / 1e9
            # write bandwidth (update last slot)
            write_t = timed_run(
                lambda: kv[:, :, :, :, -1:, :].copy_(new_kv), device_type
            )
            write_bw = new_kv.numel() * elem_size / write_t / 1e9
            print(
                f"{seq_len:<8} {format_bytes(cache_bytes):>10} "
                f"{read_bw:>8.0f}GB/s {write_bw:>8.1f}GB/s"
            )
            del kv, new_kv
        except Exception:
            break


def benchmark_autoregressive(device_type, dtype, max_mem):
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: Autoregressive Decode on {device_type.upper()} ({dtype})")
    print(f"{'CacheLen':<10} {'Step':>8} {'Tok/s':>10}")
    print(f"{'-'*30}")
    cleanup()
    elem_size = 2 if dtype != torch.float32 else 4
    for cache_len in [256, 1024, 2048]:
        cache_bytes = NUM_LAYERS * 2 * NUM_HEADS * cache_len * HEAD_DIM * elem_size
        if cache_bytes * 2 > max_mem:
            break
        try:
            q_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
                device, dtype=dtype
            )
            k_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
                device, dtype=dtype
            )
            v_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
                device, dtype=dtype
            )
            o_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
                device, dtype=dtype
            )
            hidden = torch.randn(1, 1, HIDDEN_SIZE, device=device, dtype=dtype)
            k_cache = torch.randn(
                1, NUM_HEADS, cache_len, HEAD_DIM, device=device, dtype=dtype
            )
            v_cache = torch.randn(
                1, NUM_HEADS, cache_len, HEAD_DIM, device=device, dtype=dtype
            )

            def decode_step():
                q = q_proj(hidden).view(1, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
                k_new = k_proj(hidden).view(1, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
                v_new = v_proj(hidden).view(1, 1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
                k = torch.cat([k_cache, k_new], dim=2)
                v = torch.cat([v_cache, v_new], dim=2)
                out = F.scaled_dot_product_attention(q, k, v)
                return o_proj(out.transpose(1, 2).reshape(1, 1, HIDDEN_SIZE))

            avg_t = timed_run(decode_step, device_type, min_iters=20)
            print(f"{cache_len:<10} {avg_t*1000:>7.2f}ms {1/avg_t:>9.0f}")
            del q_proj, k_proj, v_proj, o_proj, hidden, k_cache, v_cache
        except Exception:
            break


def benchmark_tts_batch_scaling(device_type, dtype, max_mem):
    """Find optimal batch size for TTS workload."""
    try:
        device = torch.device(device_type)
    except Exception:
        return
    print(f"\n{'='*60}")
    print(f"Benchmark: TTS Batch Scaling on {device_type.upper()} ({dtype})")
    print(
        f"{'Batch':<8} {'Prefill':>10} {'Decode':>10} {'Total':>10} {'Throughput':>12} {'VRAM':>10}"
    )
    print(f"{'-'*64}")
    cleanup()

    # simulate TTS workload: prefill (text encoding) + decode (audio generation)
    seq_len = 128  # ~300 char chunk tokenized
    decode_steps = 256  # ~2s of audio at 12Hz
    elem_size = 2 if dtype != torch.float32 else 4

    q_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
        device, dtype=dtype
    )
    k_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
        device, dtype=dtype
    )
    v_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
        device, dtype=dtype
    )
    o_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False).to(
        device, dtype=dtype
    )

    # extended batch sizes for high-VRAM systems
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        # check memory
        mem_needed = (
            batch_size
            * NUM_LAYERS
            * 2
            * NUM_HEADS
            * (seq_len + decode_steps)
            * HEAD_DIM
            * elem_size
        )
        if mem_needed > max_mem * 0.8:
            print(f"{batch_size:<8} {'OOM (est)':>10}")
            break

        try:
            # prefill: process full sequence
            hidden = torch.randn(
                batch_size, seq_len, HIDDEN_SIZE, device=device, dtype=dtype
            )
            k_cache = torch.zeros(
                batch_size,
                NUM_HEADS,
                seq_len + decode_steps,
                HEAD_DIM,
                device=device,
                dtype=dtype,
            )
            v_cache = torch.zeros(
                batch_size,
                NUM_HEADS,
                seq_len + decode_steps,
                HEAD_DIM,
                device=device,
                dtype=dtype,
            )

            def prefill():
                q = (
                    q_proj(hidden)
                    .view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                k = (
                    k_proj(hidden)
                    .view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                v = (
                    v_proj(hidden)
                    .view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                k_cache[:, :, :seq_len, :] = k
                v_cache[:, :, :seq_len, :] = v
                return F.scaled_dot_product_attention(q, k, v)

            prefill_t = timed_run(prefill, device_type, min_iters=5)

            # decode: single token steps (simulate 10 steps)
            hidden_dec = torch.randn(
                batch_size, 1, HIDDEN_SIZE, device=device, dtype=dtype
            )
            cache_len = seq_len + 10

            def decode_step():
                q = (
                    q_proj(hidden_dec)
                    .view(batch_size, 1, NUM_HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                k_new = (
                    k_proj(hidden_dec)
                    .view(batch_size, 1, NUM_HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                v_new = (
                    v_proj(hidden_dec)
                    .view(batch_size, 1, NUM_HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                k_cache[:, :, cache_len : cache_len + 1, :] = k_new
                v_cache[:, :, cache_len : cache_len + 1, :] = v_new
                return F.scaled_dot_product_attention(
                    q,
                    k_cache[:, :, : cache_len + 1, :],
                    v_cache[:, :, : cache_len + 1, :],
                )

            decode_t = timed_run(decode_step, device_type, min_iters=20)
            total_decode_t = decode_t * decode_steps

            total_t = prefill_t + total_decode_t
            throughput = batch_size / total_t  # chunks per second

            # get actual VRAM usage
            vram_used = ""
            if "cuda" in device_type:
                vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
                vram_used = f"{vram_mb:.0f}MB"

            print(
                f"{batch_size:<8} {prefill_t*1000:>8.1f}ms "
                f"{total_decode_t*1000:>8.0f}ms {total_t*1000:>8.0f}ms "
                f"{throughput:>10.1f}/s {vram_used:>10}"
            )

            del hidden, hidden_dec, k_cache, v_cache
            cleanup()
        except Exception as e:
            print(f"{batch_size:<8} FAILED ({e})")
            break

    del q_proj, k_proj, v_proj, o_proj


def setup_rocm_env():
    """Set environment variables for optimal ROCm performance."""
    if not (hasattr(torch.version, "hip") and torch.version.hip is not None):
        return
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    os.environ.setdefault("MIOPEN_WORKSPACE_MAX", "256000000")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.expanduser("~/.cache/miopen"))


def benchmark_qwen_tts_real(device_type, dtype, max_mem):
    """Benchmark actual qwen-tts model to verify batch processing."""
    if device_type == "cpu":
        print("Skipping real TTS benchmark on CPU (too slow)")
        return
    print(f"\n{'='*60}")
    print("Benchmark: Real Qwen-TTS Batch Processing")
    print(f"{'='*60}")

    # setup ROCm environment BEFORE loading model
    setup_rocm_env()

    # suppress repetitive HuggingFace warnings
    import transformers

    transformers.logging.set_verbosity_error()

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        print("qwen_tts not installed, skipping")
        return

    cleanup()

    # determine attention implementation
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    attn_impl = "sdpa" if is_rocm else "flash_attention_2"

    # load model
    print(f"Loading model (attn={attn_impl})...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device_type,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    if "cuda" in device_type:
        vram_model = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"Model VRAM: {vram_model:.0f}MB")

    # apply torch.compile for faster inference
    print("Compiling model...")
    try:
        model.model.talker = torch.compile(
            model.model.talker,
            mode="reduce-overhead",
            fullgraph=False,
        )
        print("Model compiled")
    except Exception as e:
        print(f"Compilation failed: {e}")

    # warmup to trigger JIT compilation
    print("Warming up...")
    with torch.inference_mode():
        try:
            _, _ = model.generate_custom_voice(
                text="Warmup.",
                language="English",
                speaker="Ryan",
                non_streaming_mode=True,
            )
            print("Warmup complete")
        except Exception as e:
            print(f"Warmup failed: {e}")

    # test different batch sizes with real synthesis
    # use ~300 char text to match typical chunk size
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a longer test sentence to better simulate real workloads. "
        "Text to speech systems perform differently with varying input lengths. "
        "Shorter inputs have higher overhead per word, while longer inputs amortize the cost. "
        "We want to measure realistic throughput for audiobook generation."
    )
    print(f"Test text: {len(test_text)} chars")
    print(f"\n{'Batch':<8} {'Time':>10} {'Audio':>10} {'VRAM':>10} {'RTF':>8}")
    print(f"{'-'*50}")

    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        try:
            texts = [test_text] * batch_size

            if "cuda" in device_type:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start = time.time()
            with torch.inference_mode():
                wavs, sr = model.generate_custom_voice(
                    text=texts,
                    language="English",
                    speaker="Ryan",
                    non_streaming_mode=True,
                )
            if "cuda" in device_type:
                torch.cuda.synchronize()
            elapsed = time.time() - start

            # calculate stats
            total_samples = sum(len(w) for w in wavs)
            audio_duration = total_samples / sr

            vram_str = ""
            if "cuda" in device_type:
                vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                vram_str = f"{vram_peak:.0f}MB"

            rtf = audio_duration / elapsed  # real-time factor
            per_sample = elapsed / batch_size

            print(
                f"{batch_size:<8} {elapsed:>8.2f}s {audio_duration:>8.1f}s "
                f"{vram_str:>10} {rtf:>7.1f}x  ({per_sample:.1f}s/sample)"
            )

        except Exception as e:
            print(f"{batch_size:<8} FAILED ({e})")
            break

    del model
    cleanup()


def benchmark_model_load(device_type, dtype, max_mem):
    print(f"\n{'='*60}")
    print("Benchmark: Model Loading (simulated 1.7B params)")
    print(f"{'='*60}")
    # use 500MB for speed, extrapolate to 3.4GB
    test_mb = 500
    scale = (1.7e9 * 2) / (test_mb * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".bin") as f:
        data = os.urandom(test_mb * 1024 * 1024)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        # mmap
        f.seek(0)
        os.posix_fadvise(f.fileno(), 0, len(data), os.POSIX_FADV_DONTNEED)
        import mmap

        start = time.time()
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        _ = mm[:]
        mm.close()
        mmap_t = time.time() - start
        print(
            f"mmap:   {test_mb / mmap_t:.0f} MB/s (full model ~{mmap_t * scale:.1f}s)"
        )
        # direct read
        f.seek(0)
        os.posix_fadvise(f.fileno(), 0, len(data), os.POSIX_FADV_DONTNEED)
        start = time.time()
        _ = f.read()
        direct_t = time.time() - start
        print(
            f"direct: {test_mb / direct_t:.0f} MB/s (full model ~{direct_t * scale:.1f}s)"
        )
        # H2D
        if device_type != "cpu" and torch.cuda.is_available():
            cleanup()
            device = torch.device(device_type)
            weights = torch.randn(test_mb * 1024 * 1024 // 2, dtype=torch.bfloat16)
            weights_pinned = weights.pin_memory()
            start = time.time()
            _ = weights_pinned.to(device)
            torch.cuda.synchronize()
            h2d_t = time.time() - start
            print(
                f"H2D:    {test_mb / h2d_t:.0f} MB/s (full model ~{h2d_t * scale:.1f}s)"
            )
            del weights, weights_pinned


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen-TTS primitives.")
    benchmarks = {
        "disk": benchmark_disk_io,
        "h2d": benchmark_h2d_transfer,
        "gemm": benchmark_gemm,
        "attention": benchmark_attention,
        "conv1d": benchmark_conv1d,
        "conv_t": benchmark_conv_transpose1d,
        "layernorm": benchmark_layernorm,
        "audio": benchmark_audio_transforms,
        "mel": benchmark_mel_spectrogram,
        "kv_cache": benchmark_kv_cache,
        "autoregressive": benchmark_autoregressive,
        "tts_batch": benchmark_tts_batch_scaling,
        "tts_real": benchmark_qwen_tts_real,
        "model_load": benchmark_model_load,
    }

    parser.add_argument(
        "benchmarks",
        nargs="*",
        default=["all"],
        help=f"Benchmarks: {', '.join(benchmarks.keys())} or 'all'",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Force specific device"
    )
    args = parser.parse_args()

    selected = (
        set(benchmarks.keys())
        if "all" in args.benchmarks
        else {b for b in args.benchmarks if b in benchmarks}
    )
    for b in args.benchmarks:
        if b != "all" and b not in benchmarks:
            print(f"Warning: unknown benchmark '{b}'")

    cpu_mem = int(get_system_memory() * 0.9)
    targets = [("cpu", torch.float32, cpu_mem)]
    if torch.cuda.is_available():
        try:
            gpu_mem = int(torch.cuda.mem_get_info(0)[1] * 0.9)
        except Exception:
            gpu_mem = 24 * 1024**3
        targets.insert(0, ("cuda", torch.bfloat16, gpu_mem))

    # device-independent benchmarks
    if "disk" in selected:
        benchmark_disk_io()
    if "model_load" in selected:
        benchmark_model_load(targets[0][0], torch.bfloat16, targets[0][2])

    for dev, dtype, mem in targets:
        if args.device and args.device != dev:
            continue
        if "h2d" in selected:
            benchmark_h2d_transfer(dev)
        for name, func in benchmarks.items():
            if name in selected and name not in ("disk", "h2d", "model_load"):
                func(dev, dtype, mem)


if __name__ == "__main__":
    main()
