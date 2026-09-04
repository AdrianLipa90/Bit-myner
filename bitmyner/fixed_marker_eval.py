from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Iterable, Sequence

from .dncrypt_marker import FeaturePredicate, MarkerGenome, raven_single_bit_population
from .gremlin_marker import deterministic_headers
from .marker_training import _wilson
from .sha256_trace import IV, K, MASK32, _compress, _rotr

FULL_COST = 128
SCHEMA = 'BITMYNER_FROZEN_MARKER_EXTERNAL_EVALUATION_V0_1'


def predicate_lattice() -> tuple[FeaturePredicate, ...]:
    genomes = raven_single_bit_population(
        stages=(1, 2),
        rounds=(0, 7, 15, 31, 47, 63),
        sources=('a', 'e', 't1', 'w'),
        bits=(0, 7, 15, 23, 31),
    )
    return tuple(g.predicates[0] for g in genomes)


def predicates_from_indices(indices: Iterable[int]) -> tuple[FeaturePredicate, ...]:
    lattice = predicate_lattice()
    idx = tuple(int(x) for x in indices)
    if not idx:
        raise ValueError('frozen genome requires at least one predicate index')
    if len(set(idx)) != len(idx):
        raise ValueError('predicate indices must be unique')
    if any(i < 0 or i >= len(lattice) for i in idx):
        raise ValueError('predicate index outside fixed lattice')
    return tuple(lattice[i] for i in sorted(idx))


def _checkpoint_cost(p: FeaturePredicate) -> int:
    return p.round + 1 if p.stage == 1 else 64 + p.round + 1


def _bitcoin_leading_zero_bits(digest: bytes) -> int:
    n = 0
    for byte in digest[::-1]:
        if byte == 0:
            n += 8
            continue
        n += 8 - byte.bit_length()
        break
    return n


def _message_schedule(block: bytes) -> list[int]:
    if len(block) != 64:
        raise ValueError('SHA-256 block must be exactly 64 bytes')
    w = [int.from_bytes(block[i:i + 4], 'big') for i in range(0, 64, 4)]
    for i in range(16, 64):
        x = w[i - 15]
        y = w[i - 2]
        s0 = _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> 3)
        s1 = _rotr(y, 17) ^ _rotr(y, 19) ^ (y >> 10)
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & MASK32)
    return w


def _source_value(source: str, *, a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, t1: int, t2: int, w: int) -> int:
    values = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g, 'h': h, 't1': t1, 't2': t2, 'w': w}
    try:
        return int(values[source])
    except KeyError as exc:
        raise ValueError(f'unsupported source: {source}') from exc


def _run_stage_marker(
    state: tuple[int, ...],
    block: bytes,
    predicates: Sequence[FeaturePredicate],
    *,
    require_final_state: bool,
) -> tuple[bool, int | None, tuple[int, ...] | None]:
    """Evaluate all-mode predicates while retaining only required checkpoints.

    Returns (match, failing checkpoint cost local to stage, final compression state if requested).
    The final state is the SHA feed-forward state, not merely the working variables.
    """
    if not predicates and not require_final_state:
        return True, None, None

    by_round: dict[int, list[FeaturePredicate]] = {}
    for p in predicates:
        by_round.setdefault(p.round, []).append(p)
    max_round = 63 if require_final_state else max(by_round, default=-1)
    w = _message_schedule(block)
    a, b, c, d, e, f, g, h = state

    for i in range(max_round + 1):
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        t1 = (h + S1 + ch + K[i] + w[i]) & MASK32
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & MASK32
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK32, c, b, a, (t1 + t2) & MASK32

        for p in by_round.get(i, ()):
            value = _source_value(
                p.source, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h,
                t1=t1, t2=t2, w=w[i],
            )
            if (value & p.mask) != p.value:
                return False, i + 1, None

    if not require_final_state:
        return True, None, None
    out = tuple((s + v) & MASK32 for s, v in zip(state, (a, b, c, d, e, f, g, h)))
    return True, None, out


def _evaluate_marker_with_midstate(
    header76: bytes,
    nonce: int,
    predicates: Sequence[FeaturePredicate],
    midstate: tuple[int, ...],
) -> tuple[bool, int]:
    stage1 = tuple(sorted((p for p in predicates if p.stage == 1), key=lambda p: (p.round, p.source, p.mask, p.value)))
    stage2 = tuple(sorted((p for p in predicates if p.stage == 2), key=lambda p: (p.round, p.source, p.mask, p.value)))

    block1 = bytearray(64)
    block1[:12] = header76[64:76]
    block1[12:16] = int(nonce).to_bytes(4, 'little')
    block1[16] = 0x80
    block1[62:64] = (80 * 8).to_bytes(2, 'big')

    need_first_digest = bool(stage2)
    ok1, fail1, first_state = _run_stage_marker(midstate, bytes(block1), stage1, require_final_state=need_first_digest)
    if not ok1:
        return False, int(fail1 or 64)
    if not stage2:
        return True, FULL_COST

    if first_state is None:
        raise RuntimeError('stage2 marker requires first SHA digest')
    first_digest = b''.join(x.to_bytes(4, 'big') for x in first_state)
    block2 = bytearray(64)
    block2[:32] = first_digest
    block2[32] = 0x80
    block2[62:64] = (32 * 8).to_bytes(2, 'big')
    ok2, fail2, _ = _run_stage_marker(IV, bytes(block2), stage2, require_final_state=False)
    if not ok2:
        return False, 64 + int(fail2 or 64)
    return True, FULL_COST


def evaluate_nonce_selected(
    header76: bytes,
    nonce: int,
    predicates: Sequence[FeaturePredicate],
    *,
    thresholds: Sequence[int] = (4, 6, 8),
) -> dict:
    if len(header76) != 76:
        raise ValueError('header76 must be exactly 76 bytes')
    if not 0 <= int(nonce) <= 0xFFFFFFFF:
        raise ValueError('nonce must fit uint32')
    if not predicates:
        raise ValueError('predicates may not be empty')
    levels = tuple(sorted(set(int(k) for k in thresholds)))
    if not levels or any(k < 1 or k > 256 for k in levels):
        raise ValueError('thresholds must be in [1,256]')

    header80 = header76 + int(nonce).to_bytes(4, 'little')
    digest = hashlib.sha256(hashlib.sha256(header80).digest()).digest()
    lz = _bitcoin_leading_zero_bits(digest)
    midstate, _ = _compress(IV, header76[:64])
    marker_match, policy_cost = _evaluate_marker_with_midstate(header76, int(nonce), predicates, midstate)
    return {
        'digest': digest,
        'leading_zero_bits': lz,
        'marker_match': marker_match,
        'policy_cost': policy_cost,
        'events': {int(k): lz >= int(k) for k in levels},
    }


def run_frozen_external_evaluation(
    *,
    predicate_indices: Sequence[int],
    header_count: int = 200,
    nonces_per_header: int = 256,
    thresholds: Sequence[int] = (4, 6, 8),
    external_seed: int = 20260904,
    training_seed: int = 616,
    training_header_count: int = 100,
    nonce_seed: int = 0xA17E5EED,
    familywise_alpha: float = 0.05,
) -> dict:
    if header_count < 1 or nonces_per_header < 1 or training_header_count < 1:
        raise ValueError('header and nonce dimensions must be positive')
    if not 0 < familywise_alpha < 1:
        raise ValueError('familywise_alpha must be in (0,1)')
    levels = tuple(sorted(set(int(k) for k in thresholds)))
    if not levels or any(k < 1 or k > 32 for k in levels):
        raise ValueError('external v0.3 thresholds must be in [1,32]')

    predicates = predicates_from_indices(predicate_indices)
    genome = MarkerGenome(predicates)
    external_headers = deterministic_headers(header_count, seed=int(external_seed))
    training_headers = deterministic_headers(training_header_count, seed=int(training_seed))
    overlap = len(set(external_headers) & set(training_headers))
    if overlap:
        raise RuntimeError(f'EXTERNAL_HEADER_OVERLAP:{overlap}')

    sample_count = header_count * nonces_per_header
    marker_n = 0
    total_policy_cost = 0
    event_n = {k: 0 for k in levels}
    marker_event_n = {k: 0 for k in levels}

    for hi, header76 in enumerate(external_headers):
        midstate, _ = _compress(IV, header76[:64])
        start = (int(nonce_seed) + hi * 0x9E3779B1) & 0xFFFFFFFF
        for j in range(nonces_per_header):
            nonce = (start + j) & 0xFFFFFFFF
            header80 = header76 + nonce.to_bytes(4, 'little')
            digest = hashlib.sha256(hashlib.sha256(header80).digest()).digest()
            lz = _bitcoin_leading_zero_bits(digest)
            marker_match, policy_cost = _evaluate_marker_with_midstate(header76, nonce, predicates, midstate)
            total_policy_cost += policy_cost
            if marker_match:
                marker_n += 1
            for k in levels:
                event = lz >= k
                if event:
                    event_n[k] += 1
                    if marker_match:
                        marker_event_n[k] += 1

    alpha_per_level = familywise_alpha / len(levels)
    rows = []
    for k in levels:
        en = event_n[k]
        men = marker_event_n[k]
        p_event = en / sample_count if sample_count else 0.0
        p_cond = men / marker_n if marker_n else 0.0
        lift = p_cond / p_event if p_event else 0.0
        cond_lo, _ = _wilson(men, marker_n, alpha_per_level) if marker_n else (0.0, 1.0)
        _, base_hi = _wilson(en, sample_count, alpha_per_level) if en else (0.0, 1.0)
        lift_lcb = cond_lo / base_hi if base_hi else 0.0
        gamma = ((men / total_policy_cost) / (en / (sample_count * FULL_COST))) if en and men and total_policy_cost else 0.0
        pass_rate = marker_n / sample_count
        avg_cost = total_policy_cost / sample_count
        gamma_lcb = FULL_COST * pass_rate * lift_lcb / avg_cost if avg_cost else 0.0
        rows.append({
            'k': k,
            'n': sample_count,
            'event_n': en,
            'marker_n': marker_n,
            'marker_event_n': men,
            'lift': lift,
            'lift_lcb': lift_lcb,
            'gamma': gamma,
            'gamma_lcb': gamma_lcb,
            'compute_fraction': total_policy_cost / (sample_count * FULL_COST),
            'alpha': alpha_per_level,
            'external_evidence_pass': bool(en >= 32 and men >= 4 and gamma_lcb > 1.0),
        })

    external_set_hash = hashlib.sha256(b''.join(external_headers)).hexdigest()
    payload = {
        'schema': SCHEMA,
        'status': 'PASS_EXTERNAL_EVALUATION_COMPLETE',
        'frozen_genome': {
            'predicate_indices': [int(x) for x in sorted(predicate_indices)],
            'genome': genome.to_dict(),
        },
        'fixture': {
            'synthetic_headers': True,
            'external_seed': int(external_seed),
            'training_seed_for_overlap_check': int(training_seed),
            'training_header_count_for_overlap_check': int(training_header_count),
            'header_count': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'sample_count': sample_count,
            'nonce_seed': int(nonce_seed),
            'header_overlap_with_training_fixture': overlap,
            'external_header_set_sha256': external_set_hash,
        },
        'policy': {
            'selection_authority': 'FROZEN_GENOME_ONLY',
            'genome_frozen_before_external_data': True,
            'retraining_after_external_data': False,
            'external_data_used_for_marker_generation': False,
            'production_policy_changed': False,
            'cryptographic_advantage_assumed': False,
            'checkpoint_storage': 'REQUIRED_PREDICATE_ROUNDS_ONLY',
            'digest_path': 'HASHLIB_SHA256D_REFERENCE',
        },
        'multiple_testing': {
            'familywise_alpha': familywise_alpha,
            'per_level_alpha': alpha_per_level,
            'correction': 'BONFERRONI_OVER_PREREGISTERED_THRESHOLDS',
        },
        'levels': rows,
        'any_external_evidence_pass': any(r['external_evidence_pass'] for r in rows),
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
