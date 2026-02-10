"""Benchmark YOLOv11 TensorRT FP16 on Jetson Orin Nano: latency, switching, power, memory."""

import argparse
import gc
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# -- Constants --

ROOT = Path(__file__).resolve().parent
MODEL_CONFIGS = {
    "nano":   {"pt": "yolo11n.pt", "engine": "yolo11n.engine"},
    "small":  {"pt": "yolo11s.pt", "engine": "yolo11s.engine"},
    "medium": {"pt": "yolo11m.pt", "engine": "yolo11m.engine"},
}
MODEL_ORDER = ["nano", "small", "medium"]
TRANSITIONS = [
    ("nano", "small"),
    ("nano", "medium"),
    ("small", "nano"),
    ("small", "medium"),
    ("medium", "nano"),
    ("medium", "small"),
]
IMGSZ = 640
OUTPUT_JSON = ROOT / "hardware_benchmark.json"

DEFAULT_PARAMS = {
    "warmup_frames": 10,
    "latency_frames": 100,
    "switch_repeats": 20,
    "sustained_seconds": 5,
    "power_measure_seconds": 5,
}
QUICK_PARAMS = {
    "warmup_frames": 3,
    "latency_frames": 20,
    "switch_repeats": 5,
    "sustained_seconds": 2,
    "power_measure_seconds": 2,
}


# -- Argument Parsing --

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv11 TensorRT FP16 on Jetson Orin Nano")
    parser.add_argument("--skip-export", action="store_true",
                        help="Reuse existing .engine files, skip TensorRT export")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced iterations: 5 repeats, 2s sustained runs")
    return parser.parse_args()


# -- Helpers --

def make_input_image():
    """Create a 640x640 random uint8 image for inference."""
    return np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)


def get_device_info():
    """Collect device/software information."""
    info = {}
    # Board name
    try:
        info["name"] = Path("/proc/device-tree/model").read_text().strip("\x00\n")
    except Exception:
        info["name"] = "unknown"
    # GPU
    try:
        info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["gpu"] = "unknown"
    # JetPack version
    try:
        text = Path("/etc/nv_tegra_release").read_text()
        info["jetpack"] = text.strip().split("\n")[0]
    except FileNotFoundError:
        try:
            out = subprocess.check_output(
                ["dpkg", "-l", "nvidia-jetpack"], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                if "nvidia-jetpack" in line and "ii" in line:
                    info["jetpack"] = line.split()[2]
                    break
            else:
                info["jetpack"] = "unknown"
        except Exception:
            info["jetpack"] = "unknown"
    info["cuda"] = getattr(torch.version, "cuda", "unknown") or "unknown"
    info["torch"] = torch.__version__
    return info


# -- TensorRT Export --

def export_engines(skip_export):
    """Export .pt models to TensorRT FP16 engines. Returns {name: engine_path}."""
    engine_paths = {}
    for name in MODEL_ORDER:
        cfg = MODEL_CONFIGS[name]
        engine_path = ROOT / cfg["engine"]

        if skip_export:
            if not engine_path.exists():
                raise FileNotFoundError(
                    f"--skip-export set but {engine_path} not found. "
                    f"Run without --skip-export first.")
            print(f"  [{name}] Reusing {engine_path}")
            engine_paths[name] = engine_path
            continue

        print(f"  [{name}] Exporting {cfg['pt']} -> TensorRT FP16 ...")
        t0 = time.perf_counter()
        model = YOLO(str(ROOT / cfg["pt"]))
        model.export(format="engine", half=True, imgsz=IMGSZ)
        elapsed = time.perf_counter() - t0
        print(f"  [{name}] Export done in {elapsed:.1f}s")

        if not engine_path.exists():
            candidates = list(ROOT.glob(f"yolo11{name[0]}*.engine"))
            if candidates:
                engine_path = candidates[0]
                print(f"  [{name}] Engine found at {engine_path}")
            else:
                raise FileNotFoundError(
                    f"Expected {engine_path} after export, not found")

        engine_paths[name] = engine_path
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return engine_paths


# -- Measurement Functions --

def measure_cold_load(engine_paths, input_img, params):
    """Cold load: time from YOLO(engine_path) to first inference complete."""
    results = {}
    for name in MODEL_ORDER:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        model = YOLO(str(engine_paths[name]))
        model.predict(input_img, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results[name] = round(elapsed_ms, 2)
        print(f"    {name}: {elapsed_ms:.1f} ms")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def measure_warm_switching(engine_paths, input_img, params):
    """Load all 3 engines, measure transition latency between pairs."""
    repeats = params["switch_repeats"]

    models = {}
    for name in MODEL_ORDER:
        models[name] = YOLO(str(engine_paths[name]))

    # Warm up each model
    for name in MODEL_ORDER:
        models[name].predict(input_img, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()

    results = {}
    for src, dst in TRANSITIONS:
        times = []
        for _ in range(repeats):
            # Complete inference on source model
            models[src].predict(input_img, imgsz=IMGSZ, verbose=False)
            torch.cuda.synchronize()

            # Measure time to complete first inference on destination
            t0 = time.perf_counter()
            models[dst].predict(input_img, imgsz=IMGSZ, verbose=False)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

        key = f"{src[0]}_to_{dst[0]}"
        results[key] = {
            "mean": round(float(np.mean(times)), 3),
            "std":  round(float(np.std(times)), 3),
        }
        print(f"    {src} -> {dst}: {np.mean(times):.2f} +/- {np.std(times):.2f} ms")

    del models
    gc.collect()
    torch.cuda.empty_cache()
    return results


def measure_inference_latency(engine_paths, input_img, params):
    """Per-model inference latency with warmup."""
    warmup = params["warmup_frames"]
    num_frames = params["latency_frames"]
    results = {}

    for name in MODEL_ORDER:
        model = YOLO(str(engine_paths[name]))

        # Warmup
        for _ in range(warmup):
            model.predict(input_img, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(num_frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(input_img, imgsz=IMGSZ, verbose=False)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

        arr = np.array(times)
        results[name] = {
            "mean": round(float(arr.mean()), 3),
            "std":  round(float(arr.std()), 3),
            "p50":  round(float(np.percentile(arr, 50)), 3),
            "p95":  round(float(np.percentile(arr, 95)), 3),
            "p99":  round(float(np.percentile(arr, 99)), 3),
        }
        print(f"    {name}: {arr.mean():.2f} +/- {arr.std():.2f} ms "
              f"(p50={np.percentile(arr, 50):.2f}, "
              f"p95={np.percentile(arr, 95):.2f}, "
              f"p99={np.percentile(arr, 99):.2f})")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def measure_max_throughput(engine_paths, input_img, params):
    """Sustained FPS over a fixed time window."""
    duration = params["sustained_seconds"]
    results = {}

    for name in MODEL_ORDER:
        model = YOLO(str(engine_paths[name]))

        # Brief warmup
        for _ in range(5):
            model.predict(input_img, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()

        # Timed loop
        count = 0
        t_start = time.perf_counter()
        while True:
            model.predict(input_img, imgsz=IMGSZ, verbose=False)
            torch.cuda.synchronize()
            count += 1
            if time.perf_counter() - t_start >= duration:
                break
        elapsed = time.perf_counter() - t_start
        fps = count / elapsed

        results[name] = round(fps, 2)
        print(f"    {name}: {fps:.1f} FPS ({count} frames in {elapsed:.2f}s)")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def measure_gpu_memory(engine_paths, input_img):
    """Measure GPU memory footprint using system-level mem_get_info()."""
    results = {}

    _, total = torch.cuda.mem_get_info(0)
    results["gpu_total"] = round(total / 1024**2, 1)

    # Per-model memory footprint
    for name in MODEL_ORDER:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        free_before, _ = torch.cuda.mem_get_info(0)

        model = YOLO(str(engine_paths[name]))
        model.predict(input_img, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()

        free_after, _ = torch.cuda.mem_get_info(0)
        footprint_mb = (free_before - free_after) / 1024**2

        results[name] = round(footprint_mb, 1)
        print(f"    {name}: {footprint_mb:.1f} MB")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # All 3 loaded simultaneously
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        free_before, _ = torch.cuda.mem_get_info(0)

        models = {}
        for name in MODEL_ORDER:
            models[name] = YOLO(str(engine_paths[name]))
        for name in MODEL_ORDER:
            models[name].predict(input_img, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()

        free_after, _ = torch.cuda.mem_get_info(0)
        all_loaded_mb = (free_before - free_after) / 1024**2

        results["all_loaded"] = round(all_loaded_mb, 1)
        print(f"    all 3 loaded: {all_loaded_mb:.1f} MB")

        del models
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        results["all_loaded"] = None
        print(f"    all 3 loaded: OOM ({str(e)[:80]})")

    gc.collect()
    torch.cuda.empty_cache()
    return results


# -- Power Monitoring --

class PowerMonitor:
    """Read Jetson power consumption. Fallback: jtop -> sysfs -> tegrastats."""

    def __init__(self):
        self.method = None
        self._jtop_cls = None
        self._sysfs_power_file = None
        if not self._init_jtop():
            if not self._init_sysfs():
                if not self._init_tegrastats():
                    print("  [power] No power monitoring available, will skip")

    def _init_jtop(self):
        try:
            from jtop import jtop
            self._jtop_cls = jtop
            self.method = "jtop"
            print("  [power] Using jtop")
            return True
        except ImportError:
            return False

    def _init_sysfs(self):
        search_roots = [
            Path("/sys/bus/i2c/drivers/ina3221x/"),
            Path("/sys/class/hwmon/"),
        ]
        for base in search_roots:
            if not base.exists():
                continue
            power_files = list(base.rglob("in_power*_input"))
            if not power_files:
                power_files = list(base.rglob("power*_input"))
            if not power_files:
                continue
            # Prefer the VDD_IN / total rail
            for pf in power_files:
                label_file = pf.parent / pf.name.replace("_input", "_label")
                if label_file.exists():
                    try:
                        label = label_file.read_text().strip()
                    except PermissionError:
                        continue
                    if "VDD" in label.upper() or "TOTAL" in label.upper():
                        self._sysfs_power_file = pf
                        self.method = "sysfs"
                        print(f"  [power] Using sysfs: {pf} ({label})")
                        return True
            # Fallback: first readable power file
            for pf in power_files:
                try:
                    pf.read_text()
                    self._sysfs_power_file = pf
                    self.method = "sysfs"
                    print(f"  [power] Using sysfs: {pf}")
                    return True
                except PermissionError:
                    continue
        return False

    def _init_tegrastats(self):
        try:
            subprocess.run(["tegrastats", "--help"],
                           capture_output=True, timeout=2)
            self.method = "tegrastats"
            print("  [power] Using tegrastats")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return False

    @property
    def available(self):
        return self.method is not None

    def read_watts(self):
        """Read instantaneous power in watts. Returns None on failure."""
        if self.method == "sysfs":
            return self._read_sysfs()
        elif self.method == "jtop":
            return self._read_jtop()
        elif self.method == "tegrastats":
            return self._read_tegrastats()
        return None

    def _read_sysfs(self):
        try:
            raw = int(self._sysfs_power_file.read_text().strip())
            return raw / 1000.0  # milliwatts -> watts
        except Exception:
            return None

    def _read_jtop(self):
        try:
            with self._jtop_cls() as jt:
                jt.start()
                time.sleep(0.1)
                power = jt.power
                if isinstance(power, dict):
                    if "tot" in power:
                        val = power["tot"]
                        w = val.get("avg", val.get("cur", 0))
                        return w / 1000.0
                    total = sum(
                        v.get("avg", v.get("cur", 0))
                        for v in power.values() if isinstance(v, dict))
                    return total / 1000.0
        except Exception:
            return None

    def _read_tegrastats(self):
        try:
            proc = subprocess.run(
                ["tegrastats", "--interval", "100"],
                capture_output=True, text=True, timeout=1)
            for line in proc.stdout.splitlines():
                m = re.search(r"VDD_\w+\s+(\d+)/(\d+)", line)
                if m:
                    return int(m.group(1)) / 1000.0
        except Exception:
            pass
        return None

    def measure_average(self, duration_seconds, interval=0.1):
        """Sample power over a duration, return mean watts."""
        if not self.available:
            return None
        samples = []
        t_end = time.perf_counter() + duration_seconds
        while time.perf_counter() < t_end:
            w = self.read_watts()
            if w is not None:
                samples.append(w)
            time.sleep(interval)
        if not samples:
            return None
        return float(np.mean(samples))


def measure_power(engine_paths, input_img, power_monitor, params):
    """Measure idle and per-model power during sustained inference."""
    if not power_monitor.available:
        print("    Power monitoring unavailable, skipping")
        return {"idle": None, "nano": None, "small": None, "medium": None}

    duration = params["power_measure_seconds"]
    results = {}

    # Idle power
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)  # Let system settle
    idle = power_monitor.measure_average(duration)
    results["idle"] = round(idle, 2) if idle is not None else None
    if idle is not None:
        print(f"    idle: {idle:.2f} W")
    else:
        print("    idle: unavailable")

    # Per-model power during inference
    for name in MODEL_ORDER:
        model = YOLO(str(engine_paths[name]))
        for _ in range(5):
            model.predict(input_img, imgsz=IMGSZ, verbose=False)
        torch.cuda.synchronize()
        time.sleep(0.5)

        # Interleave inference and power reads
        power_samples = []
        t_end = time.perf_counter() + duration
        while time.perf_counter() < t_end:
            model.predict(input_img, imgsz=IMGSZ, verbose=False)
            torch.cuda.synchronize()
            w = power_monitor.read_watts()
            if w is not None:
                power_samples.append(w)

        avg = float(np.mean(power_samples)) if power_samples else None
        results[name] = round(avg, 2) if avg is not None else None
        if avg is not None:
            print(f"    {name}: {avg:.2f} W ({len(power_samples)} samples)")
        else:
            print(f"    {name}: unavailable")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


# -- Summary --

def print_summary(results):
    """Print human-readable benchmark summary."""
    inf = results.get("inference_ms", {})
    fps = results.get("max_fps", {})
    sw = results.get("warm_switch_ms", {})
    pw = results.get("power_watts", {})
    mem = results.get("memory_mb", {})
    cold = results.get("cold_load_ms", {})

    print("\n" + "=" * 60)
    print("  HARDWARE BENCHMARK SUMMARY")
    print("=" * 60)

    # -- Inference Latency --
    print("\n=== INFERENCE LATENCY ===")
    for name in MODEL_ORDER:
        if name in inf:
            d = inf[name]
            f = fps.get(name, 0)
            print(f"  {name.capitalize():>8s}:  {d['mean']:7.2f} ms  "
                  f"({f:6.1f} FPS)  p95={d['p95']:.2f}  p99={d['p99']:.2f}")

    # -- Switching Cost --
    print("\n=== SWITCHING COST ===")
    if isinstance(sw, dict) and sw:
        for key, val in sw.items():
            src, dst = key.split("_to_")
            print(f"  {src}->{dst}:  {val['mean']:7.2f} +/- {val['std']:.2f} ms")

        n_to_m = sw.get("n_to_m")
        nano_lat = inf.get("nano", {}).get("mean")
        med_lat = inf.get("medium", {}).get("mean")
        if n_to_m and nano_lat and med_lat:
            switch_ms = n_to_m["mean"]
            slowdown = med_lat - nano_lat
            print(f"\n  Switching n->m takes {switch_ms:.1f} ms "
                  f"= {switch_ms / (1000 / 30):.1f} frames at 30fps")
            print(f"  Medium is {slowdown:.2f} ms/frame slower than Nano")
            if slowdown > 0:
                h_latency = math.ceil(switch_ms / slowdown)
                print(f"  H_latency (break-even): {h_latency} frames")
                print(f"  -> Medium must stay active >= {h_latency} frames "
                      f"to amortize the switch cost")
            print(f"  (Energy break-even requires proxy accuracy data)")
    elif isinstance(sw, str):
        print(f"  {sw}")
    else:
        print("  No switching data available")

    # -- Energy --
    print("\n=== ENERGY ===")
    idle_w = pw.get("idle")
    nano_w = pw.get("nano")
    med_w = pw.get("medium")
    if idle_w is not None:
        print(f"  Idle:    {idle_w:.2f} W")
    for name in MODEL_ORDER:
        w = pw.get(name)
        if w is not None:
            extra = f"  (idle + {w - idle_w:.2f})" if idle_w is not None else ""
            print(f"  {name.capitalize():>8s}: {w:.2f} W{extra}")

    nano_lat = inf.get("nano", {}).get("mean")
    med_lat = inf.get("medium", {}).get("mean")
    if nano_w and med_w and nano_lat and med_lat:
        nano_mj = nano_w * nano_lat  # W * ms = mJ
        med_mj = med_w * med_lat
        print(f"\n  Energy per frame:  Nano={nano_mj:.2f} mJ  "
              f"Medium={med_mj:.2f} mJ")
        print(f"  Extra cost per frame of running Medium: "
              f"{med_mj - nano_mj:.2f} mJ")
        if nano_mj > 0:
            print(f"  Medium costs {med_mj / nano_mj:.1f}x more energy "
                  f"per frame than Nano")

        # Energy cost of one n->m switch
        n_to_m = sw.get("n_to_m") if isinstance(sw, dict) else None
        if n_to_m and med_w:
            switch_energy_mj = med_w * n_to_m["mean"]
            print(f"  Energy cost of one n->m switch: "
                  f"{switch_energy_mj:.1f} mJ")
    elif all(v is None for v in [nano_w, med_w, idle_w]):
        print("  Power monitoring unavailable")

    # -- Memory --
    print("\n=== MEMORY ===")
    for name in MODEL_ORDER:
        if name in mem:
            print(f"  {name.capitalize():>8s}: {mem[name]:7.1f} MB")
    gpu_total = mem.get("gpu_total", 0)
    all_loaded = mem.get("all_loaded")
    if isinstance(all_loaded, (int, float)) and all_loaded is not None:
        print(f"  All 3:   {all_loaded:7.1f} MB  /  {gpu_total:.0f} MB total")
        can_fit = "Yes" if all_loaded < gpu_total * 0.9 else "Tight (>90%)"
        print(f"  Can all 3 TensorRT engines fit simultaneously? {can_fit}")
    elif all_loaded is None:
        print(f"  All 3:   OOM — cannot fit simultaneously")
        if cold:
            worst = max(cold.values())
            print(f"  Switching requires cold load: ~{worst:.0f} ms")
    if gpu_total:
        print(f"  GPU total: {gpu_total:.0f} MB")

    print("\n" + "=" * 60)


# -- Main --

def main():
    args = parse_args()
    params = QUICK_PARAMS if args.quick else DEFAULT_PARAMS

    if args.quick:
        print("[Quick mode: reduced iterations]")
    print(f"Parameters: {params}\n")

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a Jetson GPU.")
        return

    # Absorb CUDA context creation before any timed section
    torch.cuda.synchronize()

    # Device info
    print("=== Device Info ===")
    device_info = get_device_info()
    for k, v in device_info.items():
        print(f"  {k}: {v}")

    # Input image
    input_img = make_input_image()

    # Export engines
    print("\n=== TensorRT Export ===")
    engine_paths = export_engines(args.skip_export)

    # Cold load
    print("\n=== Cold Load Time ===")
    cold_load = measure_cold_load(engine_paths, input_img, params)

    # Inference latency
    print("\n=== Per-Frame Inference Latency ===")
    inference = measure_inference_latency(engine_paths, input_img, params)

    # Max throughput
    print("\n=== Max Throughput (FPS) ===")
    throughput = measure_max_throughput(engine_paths, input_img, params)

    # Warm switching
    print("\n=== Warm Switching Time ===")
    try:
        switching = measure_warm_switching(engine_paths, input_img, params)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        switching = f"OOM: {str(e)[:200]}"
        print(f"  Skipped: {switching}")

    # GPU memory
    print("\n=== GPU Memory ===")
    memory = measure_gpu_memory(engine_paths, input_img)

    # Power
    print("\n=== Power Consumption ===")
    power_monitor = PowerMonitor()
    power = measure_power(engine_paths, input_img, power_monitor, params)

    # Assemble and save results
    results = {
        "device_info": device_info,
        "cold_load_ms": cold_load,
        "warm_switch_ms": switching,
        "inference_ms": inference,
        "max_fps": throughput,
        "memory_mb": memory,
        "power_watts": power,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # Human-readable summary
    print_summary(results)


if __name__ == "__main__":
    main()
