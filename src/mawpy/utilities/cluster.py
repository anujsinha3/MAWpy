"""
data structure of a cluster
"""

import numpy as np

from mawpy.distance import distance


class Cluster:
    def __init__(self):
        self.pList = []
        self.center = [0, 0]
        self.radius = 0

    def add_point(self, point: tuple):
        self.pList.append(point)
        self.update_center()

    def add_points(self, list_points: list[tuple]):
        self.pList.extend(list_points)
        self.update_center()

    def update_center(self):
        self.center = np.mean([p for p in self.pList], axis=0)

    def get_distance_from_center(self, point: tuple) -> float:
        return distance(self.center[0], self.center[1], point[0], point[1])

    def get_cluster_radius(self) -> float:
        r = 0
        for p in self.pList:
            d = distance(p[0], p[1], self.center[0], self.center[1])
            if d > r:
                r = d
        return r

    def has(self, point: tuple) -> bool:
        if point in self.pList:
            return True
        return False

    def erase(self):
        self.pList = []
        self.center = [0, 0]

    def empty(self) -> bool:
        return len(self.pList) == 0
