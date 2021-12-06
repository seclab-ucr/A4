import numpy as np


class MockModel:

    def channel_axis(self):
        return 3

    def forward_one(self, x):
        prediction = np.random.rand(10)
        return prediction

    def bounds(self):
        return (0, 255)

    def num_classes(self):
        return 10


def create():
    net = MockModel()
    return net
