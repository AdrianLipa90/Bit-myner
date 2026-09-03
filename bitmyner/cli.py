from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .kernel import BitMyner, SchedulerProfile, compact_to_target, target_from_hex


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(prog="bitmyner")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--probe-count", type=int, default=2_000_000)
    b.add_argument("--holdout-count", type=int, default=16_000_000)
    b.add_argument("--holdout-repeats", type=int, default=3)
    m = sub.add_parser("mine-header")
    m.add_argument("--header76", required=True, help="76-byte Bitcoin header prefix as hex")
    tg = m.add_mutually_exclusive_group(required=True)
    tg.add_argument("--nbits", help="compact target, e.g. 1d00ffff")
    tg.add_argument("--target", help="full 256-bit target hex")
    m.add_argument("--shards", type=int, default=1)
    m.add_argument("--shard-id", type=int, default=0)
    m.add_argument("--profile", help="JSON file produced by benchmark; omitted => autotune")
    ns = ap.parse_args()

    miner = BitMyner()
    if ns.command == "benchmark":
        genesis = miner.genesis_selftest()
        profile, tuning = miner.autotune(probe_count=ns.probe_count, holdout_count=ns.holdout_count, holdout_repeats=ns.holdout_repeats)
        _print({"genesis_selftest": genesis, "profile": asdict(profile), "tuning": tuning})
        return

    header76 = bytes.fromhex(ns.header76)
    if len(header76) != 76:
        raise SystemExit("--header76 must decode to exactly 76 bytes")
    target = compact_to_target(int(ns.nbits, 16)) if ns.nbits else target_from_hex(ns.target)
    if ns.profile:
        raw = json.loads(Path(ns.profile).read_text())
        p = raw.get("profile", raw)
        profile = SchedulerProfile(**p)
    else:
        profile, _ = miner.autotune()
    _print(miner.mine_header76(header76, target, profile, total_shards=ns.shards, shard_id=ns.shard_id))


if __name__ == "__main__":
    main()
