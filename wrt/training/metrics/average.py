from __future__ import division

from .metric import Metric


class Average(Metric):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number

    @property
    def value(self):
        if self.count == 0:
            return float('inf')
        else:
            return self.sum / self.count

    def __str__(self):
        return '{:.4f}'.format(self.value)