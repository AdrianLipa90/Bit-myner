from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Iterable

DIM = 36
Q28_MASK = (1 << 28) - 1
UINT32_SPACE = 1 << 32
KAPPA = math.log(2.0) / (24.0 * math.pi)
GENESIS_HEADER76 = bytes.fromhex(
    "01000000" + "00" * 32
    + "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
    + "29ab5f49" + "ffff001d"
)
GENESIS_NONCE = 2083236893
GENESIS_HASH = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"


def canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest_obj(obj: object) -> str:
    return hashlib.blake2b(canonical_bytes(obj), digest_size=32).hexdigest()


def compact_to_target(nbits: int) -> int:
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x007FFFFF
    if nbits & 0x00800000:
        raise ValueError("negative compact target")
    return mantissa >> (8 * (3 - exponent)) if exponent <= 3 else mantissa << (8 * (exponent - 3))


def sha256d(header80: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(header80).digest()).digest()


def read_cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


@dataclass(frozen=True)
class NoemaStatus:
    status: str
    binding: str
    execution_authorized: bool
    authority: str | None
    prompt_admission_ready: bool
    vector_sha256: dict[str, str]
    phase36: tuple[float, ...]


def read_noema(root: Path = Path("/dev/shm/ciel_noema"), *, require_execution: bool = False) -> NoemaStatus:
    if not root.is_dir():
        raise RuntimeError("NOEMA_ROOT_MISSING")
    binding = (root / "ciel_binding_status").read_text().strip()
    if binding != "ACTIVE":
        raise RuntimeError(f"TETHER_NOT_ACTIVE:{binding}")
    vectors: dict[str, str] = {}
    phase36: tuple[float, ...] | None = None
    for name in ("phi", "aux_phi", "aux_feedback_phi"):
        p = root / name
        raw = p.read_bytes()
        if len(raw) != 288:
            raise RuntimeError(f"{name}_SIZE_{len(raw)}_NOT_288")
        vals = struct.unpack("<36d", raw)
        if len(vals) != DIM or not all(math.isfinite(x) for x in vals):
            raise RuntimeError(f"{name}_NOT_FINITE_36D")
        vectors[name] = hashlib.sha256(raw).hexdigest()
        if name == "phi":
            phase36 = tuple(float(x) for x in vals)
    sp = root / "session" / "startpoint.json"
    sm = root / "session" / "system_message.txt"
    if not sp.exists() or not sm.exists():
        raise RuntimeError("NOEMA_SESSION_EVIDENCE_MISSING")
    startpoint = json.loads(sp.read_text())
    if startpoint.get("status") != "VERIFIED":
        raise RuntimeError("NOEMA_STARTPOINT_NOT_VERIFIED")
    sock = Path("/run/user/1000/noema/ciel_headless.sock")
    if not sock.exists() or not sock.is_socket():
        raise RuntimeError("NOEMA_TETHER_SOCKET_MISSING")
    sup = json.loads((root / "supervisor" / "status.json").read_text())
    services = sup.get("services") or {}
    for svc in ("tether", "gremlin", "terminal36d"):
        if (services.get(svc) or {}).get("observed_status") != "ACTIVE":
            raise RuntimeError(f"NOEMA_SERVICE_NOT_ACTIVE:{svc}")
    execution_authorized = sup.get("execution_authorized") is True
    if require_execution and not execution_authorized:
        raise RuntimeError("NOEMA_EXECUTION_NOT_AUTHORIZED")
    return NoemaStatus(
        status=str(sup.get("status")),
        binding=binding,
        execution_authorized=execution_authorized,
        authority=sup.get("authority"),
        prompt_admission_ready=sup.get("prompt_admission_ready") is True,
        vector_sha256=vectors,
        phase36=phase36 or tuple(),
    )


@dataclass(frozen=True)
class PhaseNavShard:
    worker_id: int
    workers: int
    phase_slot: int
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


def phasenav_shard(worker_id: int, workers: int, phase36: Iterable[float]) -> PhaseNavShard:
    """Bijective PhaseNav assignment of worker ids to exact non-overlapping nonce intervals.

    Phase affects assignment only; it never changes SHA256d success probability.
    """
    if workers <= 0 or not 0 <= worker_id < workers:
        raise ValueError("invalid worker shard")
    packed = struct.pack("<36d", *tuple(phase36))
    seed = hashlib.blake2b(packed + b"Bit-myner/PhaseNavShard/v0.2", digest_size=32).digest()
    a = int.from_bytes(seed[:8], "little") | 1
    while math.gcd(a, workers) != 1:
        a += 2
    b = int.from_bytes(seed[8:16], "little") % workers
    slot = (a * worker_id + b) % workers
    start = (slot * UINT32_SPACE) // workers
    stop = ((slot + 1) * UINT32_SPACE) // workers
    return PhaseNavShard(worker_id, workers, slot, start, stop)


@dataclass(frozen=True)
class Candidate:
    backend: str
    threads: int
    batch_hashes: int
    genome_id: str
    route_commitment: str
    q28_schedule_id: int
    kappa: float = KAPPA


def candidate_from_phase(phase36: tuple[float, ...], backend: str, threads: int, batch_hashes: int) -> Candidate:
    raw = struct.pack("<36d", *phase36)
    seed = hashlib.shake_256(raw + backend.encode() + struct.pack("<II", threads, batch_hashes)).digest(72)
    op = list(seed[:36])
    route = sorted(range(DIM), key=lambda i: (seed[36 + i], i))
    payload = {
        "schema": "BITMYNER_DNCRYPT_PHASE36_GENOME_V0_2",
        "operator_vector": op,
        "route": route,
        "backend": backend,
        "threads": threads,
        "batch_hashes": batch_hashes,
    }
    genome_id = digest_obj(payload)
    gremlin_route = {
        "roles": ["SPIDER:correctness", "ANT:parallelism", "MANTIS:tuning", "HOUND:regression", "OWL:telemetry", "BELZEBUB:promotion"],
        "genome_id": genome_id,
    }
    route_commitment = digest_obj(gremlin_route)
    q28_schedule_id = int.from_bytes(hashlib.blake2b(canonical_bytes(payload), digest_size=8).digest(), "big") & Q28_MASK
    return Candidate(backend, threads, batch_hashes, genome_id, route_commitment, q28_schedule_id)


class NativeBackend:
    def __init__(self, name: str, library: Path):
        self.name = name
        self.library = library
        self.lib = ctypes.CDLL(str(library))
        self.U8 = ctypes.c_ubyte
        if name == "avx512x2":
            self.lanes = 32
            self._bench = self.lib.bm_avx512x2_scan_bench
            self._target = self.lib.bm_avx512x2_scan_target
        elif name == "avx512":
            self.lanes = 16
            self._bench = self.lib.bm_avx512_scan_bench
            self._target = self.lib.bm_avx512_scan_target
        elif name == "avx2":
            self.lanes = 8
            self._bench = self.lib.bm_avx2_scan_bench
            self._target = self.lib.bm_avx2_scan_target
        elif name == "shani":
            self.lanes = 1
            self._bench = self.lib.bm_scan_bench
            self._target = self.lib.bm_scan_target
        else:
            raise ValueError(name)
        self._bench.argtypes = [ctypes.POINTER(self.U8), ctypes.c_uint32, ctypes.c_uint64, ctypes.c_int]
        self._bench.restype = ctypes.c_uint64
        self._target.argtypes = [ctypes.POINTER(self.U8), ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(self.U8), ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(self.U8)]
        self._target.restype = ctypes.c_int

    def bench(self, header76: bytes, start: int, count: int, threads: int) -> int:
        a = (self.U8 * 76).from_buffer_copy(header76)
        aligned = count - (count % self.lanes)
        return int(self._bench(a, start, aligned, threads))

    def scan_target(self, header76: bytes, start: int, count: int, target: int, threads: int) -> tuple[int, bytes] | None:
        a = (self.U8 * 76).from_buffer_copy(header76)
        t = (self.U8 * 32).from_buffer_copy(target.to_bytes(32, "little"))
        found_nonce = ctypes.c_uint32()
        found_hash = (self.U8 * 32)()
        aligned = count - (count % self.lanes)
        r = self._target(a, start, aligned, t, threads, ctypes.byref(found_nonce), found_hash)
        if r:
            return int(found_nonce.value), bytes(found_hash)
        # Exact tail verification in Python, at most lanes-1 hashes.
        for n in range(start + aligned, start + count):
            h = sha256d(header76 + struct.pack("<I", n & 0xFFFFFFFF))
            if int.from_bytes(h, "little") <= target:
                return n & 0xFFFFFFFF, h
        return None


def source_root() -> Path:
    return Path(__file__).resolve().parents[1] / "native"


def available_backend_names(flags: set[str] | None = None) -> list[str]:
    flags = flags if flags is not None else read_cpu_flags()
    out: list[str] = []
    if "avx512f" in flags:
        out.extend(["avx512x2", "avx512"])
    if "avx2" in flags:
        out.append("avx2")
    if "sha_ni" in flags or "sha" in flags:
        out.append("shani")
    if "shani" not in out:
        out.append("shani")  # OpenSSL fallback remains valid even if SHA-NI is absent.
    return out


def build_backend(name: str, build_dir: Path | None = None) -> NativeBackend:
    build_dir = build_dir or (Path.cwd() / ".bitmyner-build")
    build_dir.mkdir(parents=True, exist_ok=True)
    gcc = shutil.which("gcc")
    if not gcc:
        raise RuntimeError("GCC_REQUIRED")
    source = source_root() / {"avx512x2": "sha256d_avx512_x2.c", "avx512": "sha256d_avx512.c", "avx2": "sha256d_avx2.c", "shani": "sha256d_shani.c"}[name]
    out = build_dir / f"libbitmyner_{name}.so"
    flags = ["-O3", "-mtune=native", "-fopenmp", "-fPIC", "-shared"]
    if name in {"avx512x2", "avx512"}:
        flags.insert(1, "-march=native")
    elif name == "avx2":
        flags.insert(1, "-mavx2")
    else:
        flags.insert(1, "-march=native")
    cmd = [gcc, *flags, str(source), "-o", str(out), "-lcrypto"]
    cp = subprocess.run(cmd, text=True, capture_output=True)
    if cp.returncode:
        raise RuntimeError(f"BACKEND_BUILD_FAILED:{name}\n{cp.stderr}")
    return NativeBackend(name, out)


def backend_selftest(backend: NativeBackend) -> dict[str, object]:
    target = compact_to_target(0x1D00FFFF)
    start = GENESIS_NONCE - 93
    result = backend.scan_target(GENESIS_HEADER76, start, 256, target, 1)
    ok = bool(result and result[0] == GENESIS_NONCE and result[1][::-1].hex() == GENESIS_HASH)
    if not ok:
        raise RuntimeError(f"GENESIS_SELFTEST_FAIL:{backend.name}:{result}")
    return {"backend": backend.name, "status": "PASS", "nonce": result[0], "hash": result[1][::-1].hex()}


@dataclass(frozen=True)
class BenchRecord:
    backend: str
    threads: int
    batch_hashes: int
    trials_hps: tuple[float, ...]
    median_hps: float
    max_hps: float
    jitter_cv: float
    median_batch_latency_s: float
    tau_latency_ratio: float
    promotion_eligible: bool
    score: float
    genome_id: str
    route_commitment: str
    q28_schedule_id: int


def benchmark_candidate(backend: NativeBackend, candidate: Candidate, *, repeats: int = 5, stale_budget_s: float = 0.20) -> BenchRecord:
    count = candidate.batch_hashes - (candidate.batch_hashes % backend.lanes)
    backend.bench(GENESIS_HEADER76, 0, max(backend.lanes * 4096, min(count, backend.lanes * 20000)), candidate.threads)
    samples: list[float] = []
    for r in range(repeats):
        start = (r * count) & 0xFFFFFFFF
        t0 = time.perf_counter()
        backend.bench(GENESIS_HEADER76, start, count, candidate.threads)
        dt = time.perf_counter() - t0
        samples.append(count / max(dt, 1e-12))
    med = statistics.median(samples)
    cv = statistics.pstdev(samples) / med if len(samples) > 1 and med else 0.0
    # κ acts only as a conservative instability penalty. It never alters nonce/hash semantics.
    latency = count / max(med, 1e-12)
    tau = latency / max(stale_budget_s, 1e-12)
    eligible = tau <= 1.0 and cv <= 0.15
    # κ penalizes instability; τ penalizes batches that exceed the stale-work latency budget.
    # GREMLIN promotion rejects high-jitter candidates instead of chasing transient peaks.
    score = med / (1.0 + KAPPA * cv)
    if not eligible:
        score *= 1.0 / tau
    return BenchRecord(candidate.backend, candidate.threads, count, tuple(samples), med, max(samples), cv, latency, tau, eligible, score, candidate.genome_id, candidate.route_commitment, candidate.q28_schedule_id)


def autotune(noema: NoemaStatus, *, repeats: int = 5, batch_hashes: int = 6_400_000, stale_budget_s: float = 0.20, build_dir: Path | None = None) -> dict[str, object]:
    flags = read_cpu_flags()
    names = available_backend_names(flags)
    cpu_count = max(1, os.cpu_count() or 1)
    records: list[BenchRecord] = []
    selftests: list[dict[str, object]] = []
    backend_errors: dict[str, str] = {}
    for name in names:
        try:
            backend = build_backend(name, build_dir)
            selftests.append(backend_selftest(backend))
        except Exception as exc:
            backend_errors[name] = repr(exc)
            continue
        # Search a latency/throughput lattice around the requested central batch size.
        # Keep the scalar/SHA-NI fallback grid narrow so autotuning remains bounded.
        if name in {"avx512x2", "avx512"}:
            batch_grid = sorted({max(backend.lanes * 4096, batch_hashes // 2), batch_hashes, batch_hashes * 2})
        elif name == "avx2":
            batch_grid = sorted({max(backend.lanes * 4096, batch_hashes // 2), batch_hashes})
        else:
            batch_grid = [max(backend.lanes * 4096, batch_hashes // 2)]
        for threads in range(1, cpu_count + 1):
            for bh in batch_grid:
                candidate = candidate_from_phase(noema.phase36, name, threads, bh)
                records.append(benchmark_candidate(backend, candidate, repeats=repeats, stale_budget_s=stale_budget_s))
    if not records:
        raise RuntimeError(f"NO_BACKEND_SURVIVED:{backend_errors}")
    winner = max(records, key=lambda r: (r.score, r.median_hps, -r.jitter_cv))
    payload: dict[str, object] = {
        "schema": "BITMYNER_EXTREME_AUTOTUNE_V0_2",
        "created_unix_ns": time.time_ns(),
        "noema": asdict(noema),
        "cpu_count": cpu_count,
        "cpu_flags_used": sorted(flags.intersection({"avx2", "avx512f", "sha_ni", "sha"})),
        "kappa": KAPPA,
        "tau_stale_budget_s": stale_budget_s,
        "gremlin_policy": {
            "candidate_generation": "broad",
            "promotion": "narrow_typed_deterministic_receipt_bearing",
            "roles": ["SPIDER", "ANT", "MANTIS", "HOUND", "OWL", "BELZEBUB"],
        },
        "q28_semantics": "CLASSICAL_EXACT_28_BIT_CONFIGURATION_ADDRESS_ONLY_NO_PHYSICAL_QPU_CLAIM",
        "selftests": selftests,
        "backend_errors": backend_errors,
        "records": [asdict(r) for r in records],
        "winner": asdict(winner),
    }
    payload["receipt_blake2b256"] = digest_obj(payload)
    return payload
