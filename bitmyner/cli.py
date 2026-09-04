from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .core import build_backend, compact_to_target, read_noema, phasenav_shard
from .gremlin_marker import run_experiment
from .tau_exact import autotune_tau_exact


def main() -> int:
    ap = argparse.ArgumentParser(prog="bitmyner")
    ap.add_argument("--noema-root", default="/dev/shm/ciel_noema")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    b = sub.add_parser("autotune")
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--batch-hashes", type=int, default=6_400_000)
    b.add_argument("--stale-budget-ms", type=float, default=200.0)
    b.add_argument("--output")

    mg = sub.add_parser("marker-benchmark")
    mg.add_argument("--headers", type=int, default=15)
    mg.add_argument("--nonces-per-header", type=int, default=512)
    mg.add_argument("--event-k", type=int, default=8)
    mg.add_argument("--top-k", type=int, default=8)
    mg.add_argument("--seed", type=int, default=616)
    mg.add_argument("--output")

    sh = sub.add_parser("shard")
    sh.add_argument("--worker-id", type=int, required=True)
    sh.add_argument("--workers", type=int, required=True)

    m = sub.add_parser("mine-header")
    m.add_argument("--backend", choices=["avx512x2", "avx512", "avx2", "shani"], required=True)
    m.add_argument("--threads", type=int, required=True)
    m.add_argument("--header76", required=True, help="76-byte block header prefix as hex")
    tg = m.add_mutually_exclusive_group(required=True)
    tg.add_argument("--nbits", help="compact target, e.g. 1d00ffff")
    tg.add_argument("--target", help="full 256-bit target hex")
    m.add_argument("--start", type=int, default=0)
    m.add_argument("--count", type=int, required=True)

    ns = ap.parse_args()
    root = Path(ns.noema_root)
    if ns.cmd == "status":
        print(json.dumps(read_noema(root).__dict__, indent=2, default=list))
        return 0
    if ns.cmd == "autotune":
        noema = read_noema(root, require_execution=False)
        result = autotune_tau_exact(
            noema,
            repeats=ns.repeats,
            batch_hashes=ns.batch_hashes,
            stale_budget_s=ns.stale_budget_ms / 1000.0,
        )
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS" else 3
    if ns.cmd == "marker-benchmark":
        # Research-only computation still requires a live NOEMA tether, but no
        # production mining authorization because it does not actuate a pool/job.
        read_noema(root, require_execution=False)
        result = run_experiment(
            header_count=ns.headers,
            nonces_per_header=ns.nonces_per_header,
            event_k=ns.event_k,
            top_k=ns.top_k,
            seed=ns.seed,
        )
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_EXPERIMENT_COMPLETE" else 4
    if ns.cmd == "shard":
        noema = read_noema(root, require_execution=False)
        print(json.dumps(phasenav_shard(ns.worker_id, ns.workers, noema.phase36).__dict__, indent=2))
        return 0
    if ns.cmd == "mine-header":
        read_noema(root, require_execution=True)
        header = bytes.fromhex(ns.header76)
        if len(header) != 76:
            raise SystemExit("header76 must decode to exactly 76 bytes")
        target = compact_to_target(int(ns.nbits, 16)) if ns.nbits else int(ns.target, 16)
        backend = build_backend(ns.backend)
        t0 = time.perf_counter()
        result = backend.scan_target(header, ns.start, ns.count, target, ns.threads)
        dt = time.perf_counter() - t0
        payload = {"result": None, "elapsed_s": dt, "hashes_requested": ns.count, "effective_hps": ns.count / max(dt, 1e-12)}
        if result:
            payload["result"] = {"nonce": result[0], "hash": result[1][::-1].hex()}
        print(json.dumps(payload, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
