import statistics
import numpy as np
from typing import List, Optional


class StatisticsClass:
    """
    Class for statistics analysis

    Methods:
    - __init__: initialize the class
    - count: get the count of the data
    - mean: get the mean of the data
    - std: get the standard deviation of the data
    - min_value: get the minimum value of the data
    - percentile_25: get the 25th percentile of the data
    - percentile_50: get the 50th percentile of the data
    - percentile_75: get the 75th percentile of the data
    - max_value: get the maximum value of the data
    """

    def __init__(self, data: List[float]) -> None:
        """
        Initialize the class
        """
        self.data = data

    @property
    def count(self) -> int:
        """
        Get the count of the data

        :return: count of the data
        """
        return len(self.data)

    @property
    def mean(self) -> Optional[float]:
        """
        Get the mean of the data

        :return: mean of the data
        """
        if len(self.data) == 0:
            return None
        try:
            return sum(self.data) / len(self.data)
        except (ZeroDivisionError, TypeError):
            return None

    @property
    def std(self) -> Optional[float]:
        """
        Get the standard deviation of the data

        :return: standard deviation of the data
        """
        if len(self.data) == 0:
            return None
        mean = self.mean
        try:
            return np.sqrt(sum((x - mean) ** 2 for x in self.data) / (len(self.data) - 1))
        except (ZeroDivisionError, TypeError):
            return None

    @property
    def min_value(self) -> Optional[float]:
        """
        Get the minimum value of the data

        :return: minimum value of the data
        """
        try:
            if len(self.data) == 0:
                return None
            sorted_data = sorted(self.data)
            return sorted_data[0]
        except (ZeroDivisionError, TypeError):
            return None

    @property
    def percentile_25(self) -> Optional[float]:
        """
        Get the 25th percentile of the data

        :return: 25th percentile of the data
        """
        try:
            if len(self.data) == 0:
                return None
            sorted_data = sorted(self.data)
            index = 0.25 * (len(sorted_data) - 1)
            if index.is_integer():
                return sorted_data[int(index)]
            else:
                lower_index = int(index)
                upper_index = lower_index + 1
                return (1 - (index - lower_index)) * sorted_data[lower_index] + (index - lower_index) * sorted_data[
                    upper_index]
        except (ZeroDivisionError, TypeError):
            return None

    @property
    def percentile_50(self) -> Optional[float]:
        """
        Get the 50th percentile of the data

        :return: 50th percentile of the data
        """
        try:
            if len(self.data) == 0:
                return None
            sorted_data = sorted(self.data)
            index = 0.5 * (len(sorted_data) - 1)
            if index.is_integer():
                return sorted_data[int(index)]
            else:
                lower_index = int(index)
                upper_index = lower_index + 1
                return (1 - (index - lower_index)) * sorted_data[lower_index] + (index - lower_index) * sorted_data[
                    upper_index]
        except (ZeroDivisionError, TypeError):
            return None

    @property
    def percentile_75(self) -> Optional[float]:
        """
        Get the 75th percentile of the data

        :return: 75th percentile of the data
        """
        try:
            if len(self.data) == 0:
                return None
            sorted_data = sorted(self.data)
            index = 0.75 * (len(sorted_data) - 1)
            if index.is_integer():
                return sorted_data[int(index)]
            else:
                lower_index = int(index)
                upper_index = lower_index + 1
                return (1 - (index - lower_index)) * sorted_data[lower_index] + (index - lower_index) * sorted_data[
                    upper_index]
        except (ZeroDivisionError, TypeError):
            return None

    @property
    def max_value(self) -> Optional[float]:
        """
        Get the maximum value of the data

        :return: maximum value of the data
        """
        try:
            if len(self.data) == 0:
                return None
            sorted_data = sorted(self.data)
            return sorted_data[-1]
        except (ZeroDivisionError, TypeError):
            return None

    @property
    def skewness(self) -> Optional[float]:
        """
        Get the skewness of the data

        :return: skewness of the data
        """
        try:
            if len(self.data) < 3:
                return None

            mean = self.mean
            std = self.std

            if std == 0:
                return None

            n = len(self.data)
            skewness = sum(((x - mean) / std) ** 3 for x in self.data) * n / ((n - 1) * (n - 2))
            return skewness
        except (ZeroDivisionError, TypeError):
            return None


if __name__ == "__main__":
    example_list = [np.random.randint(0, 100) for _ in range(100_00)]
    stats = StatisticsClass(example_list)
    # Compare the results with the statistics module
    print(f"Count: {stats.count} == {len(example_list)}")
    print(f"Mean: {stats.mean} == {statistics.mean(example_list)}")
    print(f"Std: {stats.std} == {statistics.stdev(example_list)}")
    print(f"Min: {stats.min_value} == {min(example_list)}")
    print(f"25th percentile: {stats.percentile_25} == {np.percentile(example_list, 25)}")
    print(f"50th percentile: {stats.percentile_50} == {np.percentile(example_list, 50)}")
    print(f"75th percentile: {stats.percentile_75} == {np.percentile(example_list, 75)}")
    print(f"Max: {stats.max_value} == {max(example_list)}")
