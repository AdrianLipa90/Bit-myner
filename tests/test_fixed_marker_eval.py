from bitmyner.dncrypt_marker import MarkerGenome
from bitmyner.fixed_marker_eval import (
    evaluate_nonce_selected,
    predicate_lattice,
    predicates_from_indices,
    run_frozen_external_evaluation,
)
from bitmyner.gremlin_marker import deterministic_headers
from bitmyner.sha256_trace import sha256d_reference, sha256d_trace


def test_predicate_lattice_preserves_v02_index_semantics():
    lattice = predicate_lattice()
    assert len(lattice) == 480
    p12, p14, p118 = predicates_from_indices((12, 14, 118))
    assert (p12.stage, p12.round, p12.source, p12.mask, p12.value) == (1, 0, 'e', 1 << 7, 0)
    assert (p14.stage, p14.round, p14.source, p14.mask, p14.value) == (1, 0, 'e', 1 << 15, 0)
    assert (p118.stage, p118.round, p118.source, p118.mask, p118.value) == (1, 15, 'w', 1 << 31, 0)


def test_selected_checkpoint_evaluator_matches_full_trace():
    header = deterministic_headers(1, seed=20260904)[0]
    predicates = predicates_from_indices((12, 14, 118))
    genome = MarkerGenome(predicates)
    for nonce in (0, 1, 17, 65537):
        compact = evaluate_nonce_selected(header, nonce, predicates, thresholds=(4, 6, 8))
        full = sha256d_trace(header, nonce)
        assert compact['digest'] == sha256d_reference(header, nonce)
        assert compact['digest'] == full.digest
        assert compact['leading_zero_bits'] == full.leading_zero_bits
        assert compact['marker_match'] == genome.matches(full)


def test_frozen_external_evaluation_is_deterministic_and_train_disjoint():
    kwargs = dict(
        predicate_indices=(12, 14, 118),
        header_count=6,
        nonces_per_header=24,
        thresholds=(2, 3, 4),
        external_seed=20260904,
        training_seed=616,
    )
    a = run_frozen_external_evaluation(**kwargs)
    b = run_frozen_external_evaluation(**kwargs)
    assert a['status'] == 'PASS_EXTERNAL_EVALUATION_COMPLETE'
    assert a['receipt_blake2b256'] == b['receipt_blake2b256']
    assert a['fixture']['sample_count'] == 144
    assert a['fixture']['header_overlap_with_training_fixture'] == 0
    assert a['policy']['genome_frozen_before_external_data'] is True
    assert a['policy']['retraining_after_external_data'] is False
    assert a['policy']['selection_authority'] == 'FROZEN_GENOME_ONLY'
    assert [x['k'] for x in a['levels']] == [2, 3, 4]


def test_fixed_evaluator_fails_closed_on_bad_indices_and_dimensions():
    try:
        predicates_from_indices((480,))
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range predicate index must fail')

    try:
        run_frozen_external_evaluation(
            predicate_indices=(), header_count=2, nonces_per_header=2, thresholds=(2,)
        )
    except ValueError:
        pass
    else:
        raise AssertionError('empty frozen genome must fail')
