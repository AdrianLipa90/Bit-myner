from __future__ import annotations

from dataclasses import dataclass, asdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time

import numpy as np

from .formalism import CandidateMeasurement, SchedulerCandidate, generate_candidates, promote_measurements, commitment

DIM = 36
UINT32_SPACE = 1 << 32
GENESIS_HEADER76 = bytes.fromhex(
    "01000000" + "00" * 32
    + "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
    + "29ab5f49" + "ffff001d"
)
GENESIS_NONCE = 2083236893
GENESIS_HASH = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"


def sha256d(header80: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(header80).digest()).digest()


def compact_to_target(nbits: int) -> int:
    exponent = (int(nbits) >> 24) & 0xFF
    mantissa = int(nbits) & 0x007FFFFF
    if int(nbits) & 0x00800000:
        raise ValueError("negative compact target")
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def target_from_hex(s: str) -> int:
    value = int(s, 16)
    if not 0 < value < (1 << 256):
        raise ValueError("target outside 256-bit positive range")
    return value


def shard_range(total_shards: int, shard_id: int) -> tuple[int, int]:
    if not 1 <= int(total_shards) <= UINT32_SPACE:
        raise ValueError("total_shards outside 1..2^32")
    if not 0 <= int(shard_id) < int(total_shards):
        raise ValueError("shard_id outside range")
    start = (UINT32_SPACE * int(shard_id)) // int(total_shards)
    end = (UINT32_SPACE * (int(shard_id) + 1)) // int(total_shards)
    return start, end


def _read_f64_36(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) != 288:
        raise RuntimeError(f"{path.name}_SIZE_{len(raw)}_NOT_288")
    vec = np.frombuffer(raw, dtype="<f8")
    if vec.shape != (36,) or not np.isfinite(vec).all():
        raise RuntimeError(f"{path.name}_NOT_FINITE_36D")
    return vec.copy()


def noema_gate(root: Path = Path("/dev/shm/ciel_noema"), *, require_execution: bool = False) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError("NOEMA_ROOT_MISSING")
    binding = (root / "ciel_binding_status").read_text().strip()
    if binding != "ACTIVE":
        raise RuntimeError(f"TETHER_NOT_ACTIVE:{binding}")
    vectors = {}
    for name in ("phi", "aux_phi", "aux_feedback_phi"):
        vec = _read_f64_36(root / name)
        vectors[name] = hashlib.sha256(vec.astype("<f8").tobytes()).hexdigest()
    startpoint_path = root / "session" / "startpoint.json"
    system_message = root / "session" / "system_message.txt"
    if not startpoint_path.exists() or not system_message.exists():
        raise RuntimeError("NOEMA_SESSION_EVIDENCE_MISSING")
    startpoint = json.loads(startpoint_path.read_text())
    if startpoint.get("status") != "VERIFIED":
        raise RuntimeError("NOEMA_STARTPOINT_NOT_VERIFIED")
    sock = Path("/run/user/1000/noema/ciel_headless.sock")
    if not sock.exists() or not sock.is_socket():
        raise RuntimeError("NOEMA_TETHER_SOCKET_MISSING")
    supervisor_path = root / "supervisor" / "status.json"
    supervisor = json.loads(supervisor_path.read_text()) if supervisor_path.exists() else {}
    services = supervisor.get("services", {})
    for service in ("tether", "gremlin", "terminal36d"):
        if services.get(service, {}).get("observed_status") != "ACTIVE":
            raise RuntimeError(f"NOEMA_SERVICE_NOT_ACTIVE:{service}")
    if require_execution and supervisor.get("execution_authorized") is not True:
        raise RuntimeError("NOEMA_EXECUTION_NOT_AUTHORIZED")
    return {
        "status": "PASS",
        "binding": binding,
        "authority": supervisor.get("authority"),
        "execution_authorized": supervisor.get("execution_authorized"),
        "prompt_admission_ready": supervisor.get("prompt_admission_ready"),
        "resource_scheduler": supervisor.get("resource_scheduler"),
        "startpoint_packet_sha256": startpoint.get("packet_sha256"),
        "vectors": vectors,
    }


def cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags") or line.startswith("Features"):
            return set(line.split(":", 1)[1].strip().split())
    return set()


@dataclass(frozen=True, slots=True)
class SchedulerProfile:
    kernel_variant: str
    threads: int
    batch_size: int
    median_hps: float
    candidate_receipt_sha256: str
    tuning_receipt_sha256: str


class NativeKernel:
    def __init__(self, library: Path):
        self.library = Path(library)
        self.lib = ctypes.CDLL(str(self.library))
        u8p = ctypes.POINTER(ctypes.c_ubyte)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        self.lib.bitmyner_hash_one.argtypes = [u8p, ctypes.c_uint32, u8p]
        self.lib.bitmyner_hash_one.restype = ctypes.c_int
        for name in ("bitmyner_scan_mt_mask", "bitmyner_scan_mt_mask32"):
            fn = getattr(self.lib, name)
            fn.argtypes = [u8p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int, u32p, u32p]
            fn.restype = ctypes.c_int

    @staticmethod
    def build(source: Path, output: Path) -> Path:
        if "avx512f" not in cpu_flags():
            raise RuntimeError("AVX512F_REQUIRED_FOR_NATIVE_BACKEND")
        gcc = shutil.which("gcc")
        if not gcc:
            raise RuntimeError("GCC_REQUIRED_FOR_NATIVE_BACKEND")
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            gcc, "-O3", "-funroll-loops", "-frename-registers", "-fPIC", "-shared",
            "-march=native", "-mtune=native", "-mavx512f", "-mavx512bw", "-fopenmp",
            "-o", str(output), str(source),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"NATIVE_BUILD_FAILED\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}")
        return output

    def hash_one(self, header76: bytes, nonce: int) -> bytes:
        if len(header76) != 76:
            raise ValueError("header76 must be exactly 76 bytes")
        inp = (ctypes.c_ubyte * 76).from_buffer_copy(header76)
        out = (ctypes.c_ubyte * 32)()
        rc = self.lib.bitmyner_hash_one(inp, ctypes.c_uint32(nonce), out)
        if rc:
            raise RuntimeError(f"NATIVE_HASH_ONE_FAILED:{rc}")
        return bytes(out)

    def scan(self, variant: str, header76: bytes, start_nonce: int, count: int, target_hi: int, threads: int) -> tuple[int | None, int]:
        if len(header76) != 76:
            raise ValueError("header76 must be exactly 76 bytes")
        if not 0 <= start_nonce < UINT32_SPACE:
            raise ValueError("start_nonce outside uint32")
        if not 0 < count <= 0xFFFFFFFF:
            raise ValueError("count outside uint32 positive range")
        inp = (ctypes.c_ubyte * 76).from_buffer_copy(header76)
        found = ctypes.c_uint32()
        checked = ctypes.c_uint32()
        if variant == "avx512x32":
            fn = self.lib.bitmyner_scan_mt_mask32
        elif variant == "avx512x16":
            fn = self.lib.bitmyner_scan_mt_mask
        else:
            raise ValueError(f"unknown native variant: {variant}")
        rc = fn(inp, ctypes.c_uint32(start_nonce), ctypes.c_uint32(count), ctypes.c_uint64(target_hi), ctypes.c_int(threads), ctypes.byref(found), ctypes.byref(checked))
        return (int(found.value) if rc else None, int(checked.value))


def _default_source() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "sha256d_avx512.c"


def _default_library() -> Path:
    cache = Path(os.environ.get("BITMYNER_CACHE", Path.home() / ".cache" / "bitmyner"))
    return cache / "libbitmyner_avx512.so"


class BitMyner:
    def __init__(self, *, noema_root: Path = Path("/dev/shm/ciel_noema"), source: Path | None = None, library: Path | None = None):
        self.noema_root = Path(noema_root)
        self.source = Path(source) if source else _default_source()
        self.library = Path(library) if library else _default_library()
        if not self.library.exists() or self.library.stat().st_mtime_ns < self.source.stat().st_mtime_ns:
            NativeKernel.build(self.source, self.library)
        self.native = NativeKernel(self.library)

    def genesis_selftest(self) -> dict:
        digest = self.native.hash_one(GENESIS_HEADER76, GENESIS_NONCE)
        display = digest[::-1].hex()
        if display != GENESIS_HASH:
            raise RuntimeError(f"GENESIS_SELFTEST_FAIL:{display}")
        reference = sha256d(GENESIS_HEADER76 + struct.pack("<I", GENESIS_NONCE))
        if digest != reference:
            raise RuntimeError("GENESIS_REFERENCE_MISMATCH")
        return {"status": "PASS", "nonce": GENESIS_NONCE, "hash": display}

    def _benchmark_candidate(self, candidate: SchedulerCandidate, *, count: int, repeats: int) -> CandidateMeasurement:
        samples: list[float] = []
        count = max(32, int(count) - (int(count) % 32))
        for _ in range(max(1, int(repeats))):
            t0 = time.perf_counter()
            found, checked = self.native.scan(candidate.kernel_variant, GENESIS_HEADER76, 0, count, 0, candidate.threads)
            elapsed = time.perf_counter() - t0
            if found is not None or checked != count:
                raise RuntimeError("BENCHMARK_SCAN_INVARIANT_FAIL")
            samples.append(count / max(elapsed, 1e-12))
        return CandidateMeasurement(candidate, tuple(samples), statistics.median(samples), min(samples), max(samples))

    def autotune(self, *, probe_count: int = 2_000_000, holdout_count: int = 16_000_000, probe_repeats: int = 1, holdout_repeats: int = 3) -> tuple[SchedulerProfile, dict]:
        gate = noema_gate(self.noema_root)
        phase36 = _read_f64_36(self.noema_root / "phi")
        cpu = max(1, int((gate.get("resource_scheduler") or {}).get("cpu_count") or os.cpu_count() or 1))
        candidates = generate_candidates(phase36, cpu)
        unique: dict[tuple[str, int], SchedulerCandidate] = {}
        for candidate in candidates:
            unique.setdefault((candidate.kernel_variant, candidate.threads), candidate)
        probes = [self._benchmark_candidate(candidate, count=probe_count, repeats=probe_repeats) for candidate in unique.values()]
        top = sorted(probes, key=lambda measurement: measurement.median_hps, reverse=True)[: min(3, len(probes))]
        holdouts = [self._benchmark_candidate(measurement.candidate, count=holdout_count, repeats=holdout_repeats) for measurement in top]
        winner = promote_measurements(holdouts)
        pressure = str((gate.get("resource_scheduler") or {}).get("pressure") or "UNKNOWN")
        if pressure == "LOW":
            batch_size = 1 << 24
        elif pressure == "MEDIUM":
            batch_size = 1 << 22
        else:
            batch_size = 1 << 20
        candidate_receipt = winner.receipt
        tuning = {
            "schema": "BITMYNER_NOEMA_DNCRYPT_GREMLIN_TUNING_V2",
            "created_unix_ns": time.time_ns(),
            "noema": gate,
            "cpu_count": cpu,
            "candidate_space": len(candidates),
            "q28_address_capacity": 64,
            "q28_hot_path": False,
            "q28_role": "SCHEDULER_ADDRESS_ONLY",
            "phase_nav_role": "LIVE_36D_SCHEDULER_STATE",
            "dncrypt_role": "DETERMINISTIC_CANDIDATE_GENERATION_AND_ORDER",
            "gremlin_roles": ["SPIDER:correctness", "ANT:parallelism", "MANTIS:tuning", "BELZEBUB:promotion"],
            "success_probability_claim": "UNCHANGED_BY_SCHEDULER",
            "probe_measurements": [measurement.receipt for measurement in probes],
            "holdout_measurements": [measurement.receipt for measurement in holdouts],
            "winner": candidate_receipt,
            "selected_batch_size": batch_size,
            "resource_pressure": pressure,
        }
        tuning["receipt_sha256"] = commitment(tuning)
        profile = SchedulerProfile(
            winner.candidate.kernel_variant,
            winner.candidate.threads,
            batch_size,
            winner.median_hps,
            candidate_receipt["receipt_sha256"],
            tuning["receipt_sha256"],
        )
        return profile, tuning

    def mine_header76(self, header76: bytes, target: int, profile: SchedulerProfile, *, total_shards: int = 1, shard_id: int = 0) -> dict:
        gate = noema_gate(self.noema_root, require_execution=True)
        if len(header76) != 76:
            raise ValueError("header76 must be exactly 76 bytes")
        if not 0 < int(target) < (1 << 256):
            raise ValueError("target outside 256-bit positive range")
        start, end = shard_range(total_shards, shard_id)
        target_hi = int(target) >> 192
        cursor = start
        checked_total = 0
        started = time.perf_counter()
        while cursor < end:
            noema_gate(self.noema_root)
            count = min(profile.batch_size, end - cursor, 0xFFFFFFFF)
            candidate, checked = self.native.scan(profile.kernel_variant, header76, cursor, count, target_hi, profile.threads)
            checked_total += checked
            if candidate is not None:
                digest = sha256d(header76 + struct.pack("<I", candidate))
                if int.from_bytes(digest, "little") <= int(target):
                    elapsed = time.perf_counter() - started
                    result = {
                        "schema": "BITMYNER_FOUND_V2",
                        "status": "FOUND",
                        "nonce": candidate,
                        "hash_display": digest[::-1].hex(),
                        "hashes_checked": checked_total,
                        "elapsed_s": elapsed,
                        "hashes_per_second": checked_total / max(elapsed, 1e-12),
                        "shard": [shard_id, total_shards],
                        "profile": asdict(profile),
                        "noema": gate,
                    }
                    result["receipt_sha256"] = commitment(result)
                    return result
            cursor += count
        elapsed = time.perf_counter() - started
        result = {
            "schema": "BITMYNER_EXHAUSTED_V2",
            "status": "EXHAUSTED_SHARD",
            "hashes_checked": checked_total,
            "elapsed_s": elapsed,
            "hashes_per_second": checked_total / max(elapsed, 1e-12),
            "shard": [shard_id, total_shards],
            "profile": asdict(profile),
            "noema": gate,
        }
        result["receipt_sha256"] = commitment(result)
        return result
