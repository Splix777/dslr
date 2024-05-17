import statistics
import unittest

import numpy as np
import scipy.stats as scipy_stats

from statistics_class import StatisticsClass


class TestStatisticsClass(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = [np.random.randint(0, 100) for _ in range(100_00)]
        self.stats = StatisticsClass(self.data)

    def test_count(self):
        self.assertEqual(self.stats.count, len(self.data))

    def test_mean(self):
        self.assertAlmostEqual(
            self.stats.mean, statistics.mean(self.data), places=10
        )

    def test_std(self):
        self.assertAlmostEqual(
            self.stats.std, statistics.stdev(self.data), places=10
        )

    def test_min_value(self):
        self.assertEqual(self.stats.min_value, min(self.data))

    def test_percentile_25(self):
        self.assertAlmostEqual(
            self.stats.percentile_25, np.percentile(self.data, 25), places=10
        )

    def test_percentile_50(self):
        self.assertAlmostEqual(
            self.stats.percentile_50, np.percentile(self.data, 50), places=10
        )

    def test_percentile_75(self):
        self.assertAlmostEqual(
            self.stats.percentile_75, np.percentile(self.data, 75), places=10
        )

    def test_max_value(self):
        self.assertEqual(self.stats.max_value, max(self.data))

    def test_skewness(self):
        stats_skewness = self.stats.skewness
        nd_array = np.array(self.data)
        scipy_skewness = scipy_stats.skew(nd_array)

        self.assertAlmostEqual(
            float(stats_skewness), float(scipy_skewness), places=5
        )

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
