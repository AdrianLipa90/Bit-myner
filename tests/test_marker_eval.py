from bitmyner.marker_eval import MarkerObservation, lift_report, belzebub_promotion_verdict


def test_positive_control_has_large_lift():
    rows = []
    for i in range(10000):
        marker = (i % 10) < 2
        event = (i % 10) == 0
        rows.append(MarkerObservation(str(i // 100), marker, event))
    r = lift_report(rows)
    assert r.lift == 5.0
    assert r.lift_lcb > 4.0


def test_belzebub_requires_holdout_and_validation_lcb():
    evaluation = {
        'reports': {
            'train': {'lift_lcb': 2.0, 'marker_n': 100, 'event_n': 10},
            'validation': {'lift_lcb': 1.2, 'marker_n': 100, 'event_n': 10},
            'holdout': {'lift_lcb': 1.1, 'marker_n': 100, 'event_n': 10},
        }
    }
    assert belzebub_promotion_verdict(evaluation)['promote']
    evaluation['reports']['holdout']['lift_lcb'] = 0.99
    assert not belzebub_promotion_verdict(evaluation)['promote']
