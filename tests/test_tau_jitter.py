from fractions import Fraction
import unittest

from bitmyner.tau_jitter import tau_modulation_from_latencies_ns


class TauJitterModulationTests(unittest.TestCase):
    def test_zero_jitter_is_identity(self) -> None:
        tau = tau_modulation_from_latencies_ns([100_000_000] * 5, 0.2)
        self.assertEqual(tau.tau_jitter, Fraction(1, 1))
        self.assertEqual(tau.tau_latency, Fraction(1, 1))
        self.assertEqual(tau.tau_scalar, Fraction(1, 1))

    def test_jitter_is_exact_positive_rational(self) -> None:
        tau = tau_modulation_from_latencies_ns(
            [100_000_000, 100_000_000, 200_000_000, 200_000_000],
            0.2,
        )
        # median=150 ms, mean absolute jitter=50 ms => 150/(150+50)=3/4.
        self.assertEqual(tau.tau_jitter, Fraction(3, 4))
        self.assertEqual(tau.tau_latency, Fraction(1, 1))
        self.assertEqual(tau.tau_scalar, Fraction(3, 4))

    def test_jitter_scalar_is_scale_invariant(self) -> None:
        a = tau_modulation_from_latencies_ns([10, 10, 20, 20], 1.0)
        b = tau_modulation_from_latencies_ns([10_000, 10_000, 20_000, 20_000], 1.0)
        self.assertEqual(a.tau_jitter, b.tau_jitter)

    def test_latency_and_jitter_compose_multiplicatively(self) -> None:
        tau = tau_modulation_from_latencies_ns(
            [2_000_000, 2_000_000, 4_000_000, 4_000_000],
            0.002,
        )
        # median=3 ms, mean absolute jitter=1 ms -> s_jitter=3/4.
        # stale budget=2 ms -> s_latency=2/3. Composition is exactly 1/2.
        self.assertEqual(tau.tau_jitter, Fraction(3, 4))
        self.assertEqual(tau.tau_latency, Fraction(2, 3))
        self.assertEqual(tau.tau_scalar, Fraction(1, 2))

    def test_invalid_latency_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            tau_modulation_from_latencies_ns([100, 0, 100], 0.2)
        with self.assertRaises(ValueError):
            tau_modulation_from_latencies_ns([], 0.2)
        with self.assertRaises(ValueError):
            tau_modulation_from_latencies_ns([100, 100], 0.0)


if __name__ == "__main__":
    unittest.main()
