from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .core import build_backend, compact_to_target, read_noema, phasenav_shard
from .gremlin_marker import run_experiment
from .gremlin_marker_v2 import run_experiment_v2
from .tau_exact import autotune_tau_exact
from .marker_training_v4 import run_training_1000_v4
from .marker_training_v6 import run_island_training_v6
from .marker_training_v7 import run_adaptive_island_training_v7
from .marker_training_v8 import run_multi_pilot_island_training_v8
from .marker_bridge_v9 import run_bridge_recombination_v9
from .marker_training_v10 import run_nested_crossfit_training_v10
from .marker_training_v11 import run_nested_crossfit_training_v11


def _parse_thresholds(text: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(x.strip()) for x in text.split(',') if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError('thresholds must be comma-separated integers') from exc
    if not levels:
        raise argparse.ArgumentTypeError('at least one threshold is required')
    return levels


def _parse_int_tuple(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(x.strip()) for x in text.split(',') if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from exc
    if not values:
        raise argparse.ArgumentTypeError('at least one integer is required')
    return values


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

    mg2 = sub.add_parser("marker-benchmark-v2")
    mg2.add_argument("--headers", type=int, default=21)
    mg2.add_argument("--nonces-per-header", type=int, default=512)
    mg2.add_argument("--thresholds", type=_parse_thresholds, default=(6, 8, 10, 12))
    mg2.add_argument("--seed-top-k", type=int, default=16)
    mg2.add_argument("--pair-limit", type=int, default=64)
    mg2.add_argument("--top-k", type=int, default=8)
    mg2.add_argument("--seed", type=int, default=616)
    mg2.add_argument("--output")

    mt4 = sub.add_parser("marker-train-v4")
    mt4.add_argument("--iterations", type=int, default=1000)
    mt4.add_argument("--headers", type=int, default=100)
    mt4.add_argument("--nonces-per-header", type=int, default=128)
    mt4.add_argument("--thresholds", type=_parse_thresholds, default=(4, 6, 8))
    mt4.add_argument("--population", type=int, default=64)
    mt4.add_argument("--offspring", type=int, default=8)
    mt4.add_argument("--max-predicates", type=int, default=3)
    mt4.add_argument("--seed", type=int, default=616)
    mt4.add_argument("--output")

    mi6 = sub.add_parser("marker-islands-v6")
    mi6.add_argument("--meta", required=True, help="v0.5 multi-seed meta receipt JSON")
    mi6.add_argument("--iterations", type=int, default=1000)
    mi6.add_argument("--headers", type=int, default=100)
    mi6.add_argument("--nonces-per-header", type=int, default=128)
    mi6.add_argument("--thresholds", type=_parse_thresholds, default=(4, 6, 8))
    mi6.add_argument("--seeds", default="919,1020,1121")
    mi6.add_argument("--prior-strength", type=float, default=2.0)
    mi6.add_argument("--output")

    mt7 = sub.add_parser("marker-train-v7")
    mt7.add_argument("--meta", required=True, help="v0.5 multi-seed meta receipt JSON")
    mt7.add_argument("--iterations", type=int, default=1000)
    mt7.add_argument("--headers", type=int, default=100)
    mt7.add_argument("--nonces-per-header", type=int, default=128)
    mt7.add_argument("--thresholds", type=_parse_thresholds, default=(4, 6, 8))
    mt7.add_argument("--seeds", type=_parse_int_tuple, default=(1919, 2020, 2121))
    mt7.add_argument("--pilot-iterations", type=int, default=120)
    mt7.add_argument("--calibration-seed", type=int, default=1717)
    mt7.add_argument("--output")

    mt8 = sub.add_parser("marker-train-v8")
    mt8.add_argument("--meta", required=True, help="v0.5 multi-seed meta receipt JSON")
    mt8.add_argument("--iterations", type=int, default=1000)
    mt8.add_argument("--headers", type=int, default=100)
    mt8.add_argument("--nonces-per-header", type=int, default=128)
    mt8.add_argument("--thresholds", type=_parse_thresholds, default=(4, 6, 8))
    mt8.add_argument("--seeds", type=_parse_int_tuple, default=(2222, 2323, 2424))
    mt8.add_argument("--pilot-seeds", type=_parse_int_tuple, default=(1317, 1417, 1517))
    mt8.add_argument("--pilot-iterations", type=int, default=80)
    mt8.add_argument("--output")

    mb9 = sub.add_parser("marker-bridge-v9")
    mb9.add_argument("--meta", required=True, help="multi-seed meta receipt JSON")
    mb9.add_argument("--bridge-seed", type=int, default=2626)
    mb9.add_argument("--headers", type=int, default=80)
    mb9.add_argument("--nonces-per-header", type=int, default=96)
    mb9.add_argument("--thresholds", type=_parse_thresholds, default=(4, 6, 8))
    mb9.add_argument("--top-singles-per-family", type=int, default=3)
    mb9.add_argument("--pair-limit", type=int, default=32)
    mb9.add_argument("--top-k", type=int, default=6)
    mb9.add_argument("--output")

    mt10 = sub.add_parser("marker-train-v10")
    mt10.add_argument("--iterations", type=int, default=1000)
    mt10.add_argument("--headers", type=int, default=100)
    mt10.add_argument("--nonces-per-header", type=int, default=128)
    mt10.add_argument("--thresholds", type=_parse_thresholds, default=(4, 6, 8))
    mt10.add_argument("--seeds", type=_parse_int_tuple, default=(3030, 3131, 3232))
    mt10.add_argument("--population", type=int, default=64)
    mt10.add_argument("--offspring", type=int, default=8)
    mt10.add_argument("--max-predicates", type=int, default=3)
    mt10.add_argument("--discovery-fraction", type=float, default=0.72)
    mt10.add_argument("--output")

    mt11 = sub.add_parser("marker-train-v11")
    mt11.add_argument("--iterations", type=int, default=1000)
    mt11.add_argument("--headers", type=int, default=160)
    mt11.add_argument("--nonces-per-header", type=int, default=128)
    mt11.add_argument("--thresholds", type=_parse_thresholds, default=(4, 6, 8))
    mt11.add_argument("--seeds", type=_parse_int_tuple, default=(4040, 4141, 4242))
    mt11.add_argument("--population", type=int, default=64)
    mt11.add_argument("--offspring", type=int, default=8)
    mt11.add_argument("--max-predicates", type=int, default=3)
    mt11.add_argument("--discovery-fraction", type=float, default=0.72)
    mt11.add_argument("--max-cohorts", type=int, default=4)
    mt11.add_argument("--min-expected-events-per-cohort", type=float, default=8.0)
    mt11.add_argument("--output")

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
        result = autotune_tau_exact(noema, repeats=ns.repeats, batch_hashes=ns.batch_hashes, stale_budget_s=ns.stale_budget_ms / 1000.0)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS" else 3
    if ns.cmd == "marker-benchmark":
        read_noema(root, require_execution=False)
        result = run_experiment(header_count=ns.headers, nonces_per_header=ns.nonces_per_header, event_k=ns.event_k, top_k=ns.top_k, seed=ns.seed)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_EXPERIMENT_COMPLETE" else 4
    if ns.cmd == "marker-benchmark-v2":
        read_noema(root, require_execution=False)
        result = run_experiment_v2(header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, seed_top_k=ns.seed_top_k, pair_limit=ns.pair_limit, final_top_k=ns.top_k, seed=ns.seed)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_EXPERIMENT_COMPLETE" else 5
    if ns.cmd == "marker-train-v4":
        read_noema(root, require_execution=False)
        result = run_training_1000_v4(iterations=ns.iterations, header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, population_size=ns.population, offspring_per_iteration=ns.offspring, max_predicates=ns.max_predicates, seed=ns.seed, checkpoint_every=max(1, ns.iterations // 10))
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_TRAINING_COMPLETE" else 6
    if ns.cmd == "marker-islands-v6":
        read_noema(root, require_execution=False)
        meta = json.loads(Path(ns.meta).read_text())
        try:
            seeds = tuple(int(x.strip()) for x in ns.seeds.split(',') if x.strip())
        except ValueError as exc:
            raise SystemExit("--seeds must be comma-separated integers") from exc
        result = run_island_training_v6(meta, iterations=ns.iterations, seeds=seeds, header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, focused_prior_strength=ns.prior_strength)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_ISLAND_TRAINING_COMPLETE" else 7
    if ns.cmd == "marker-train-v7":
        read_noema(root, require_execution=False)
        meta = json.loads(Path(ns.meta).read_text())
        result = run_adaptive_island_training_v7(meta, iterations=ns.iterations, seeds=ns.seeds, header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, pilot_iterations=ns.pilot_iterations, calibration_seed=ns.calibration_seed)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_ADAPTIVE_ISLAND_TRAINING_COMPLETE" else 8
    if ns.cmd == "marker-train-v8":
        read_noema(root, require_execution=False)
        meta = json.loads(Path(ns.meta).read_text())
        result = run_multi_pilot_island_training_v8(meta, iterations=ns.iterations, full_seeds=ns.seeds, header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, pilot_iterations=ns.pilot_iterations, pilot_seeds=ns.pilot_seeds)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_MULTI_PILOT_ISLAND_TRAINING_COMPLETE" else 9
    if ns.cmd == "marker-bridge-v9":
        read_noema(root, require_execution=False)
        meta = json.loads(Path(ns.meta).read_text())
        result = run_bridge_recombination_v9(meta, bridge_seed=ns.bridge_seed, header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, top_singles_per_family=ns.top_singles_per_family, pair_limit=ns.pair_limit, final_top_k=ns.top_k)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_BRIDGE_RECOMBINATION_COMPLETE" else 10
    if ns.cmd == "marker-train-v10":
        read_noema(root, require_execution=False)
        result = run_nested_crossfit_training_v10(iterations=ns.iterations, seeds=ns.seeds, header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, population_size=ns.population, offspring_per_iteration=ns.offspring, max_predicates=ns.max_predicates, train_discovery_fraction=ns.discovery_fraction)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_NESTED_CROSSFIT_TRAINING_COMPLETE" else 11
    if ns.cmd == "marker-train-v11":
        read_noema(root, require_execution=False)
        result = run_nested_crossfit_training_v11(iterations=ns.iterations, seeds=ns.seeds, header_count=ns.headers, nonces_per_header=ns.nonces_per_header, thresholds=ns.thresholds, population_size=ns.population, offspring_per_iteration=ns.offspring, max_predicates=ns.max_predicates, train_discovery_fraction=ns.discovery_fraction, requested_max_cohorts=ns.max_cohorts, min_expected_events_per_cohort=ns.min_expected_events_per_cohort)
        text = json.dumps(result, indent=2)
        if ns.output:
            Path(ns.output).write_text(text + "\n")
        print(text)
        return 0 if result.get("status") == "PASS_NESTED_CROSSFIT_V11_COMPLETE" else 12
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
