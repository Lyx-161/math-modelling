import numpy as np

class BaseParameterizer:
    def __init__(self, points):
        self.points = points

    def parameterize(self):
        raise NotImplementedError("Subclasses must implement the parameterize method")

class UniformParameterizer(BaseParameterizer):
    def parameterize(self):
        n = len(self.points)
        t = np.linspace(0, 1, n)
        return t

class ChordalParameterizer(BaseParameterizer):
    def parameterize(self):
        points = np.array(self.points)
        n = len(points)
        t = np.zeros(n)
        t[0] = 0
        total_length = 0.0

        for i in range(1, n):
            dist = np.linalg.norm(points[i] - points[i-1])
            total_length += dist
            t[i] = total_length

        # Normalize to [0, 1]
        t = t / t[-1]
        return t

class CentripetalParameterizer(BaseParameterizer):
    def parameterize(self):
        points = np.array(self.points)
        n = len(points)
        t = np.zeros(n)
        t[0] = 0
        total_length = 0.0

        for i in range(1, n):
            dist = np.linalg.norm(points[i] - points[i-1])
            total_length += np.sqrt(dist)
            t[i] = total_length

        # Normalize to [0, 1]
        t = t / t[-1]
        return t

class FoleyNielsenParameterizer(BaseParameterizer):
    def parameterize(self):
        points = np.array(self.points)
        n = len(points)
        t = np.zeros(n)
        t[0] = 0
        total_length = 0.0

        for i in range(1, n):
            dist = np.linalg.norm(points[i] - points[i-1])
            # 计算角度
            if i > 1:
                v1 = points[i-1] - points[i-2]
                v2 = points[i] - points[i-1]
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            else:
                angle = 0.0
            # 结合弦长和角度
            total_length += dist * (1 + angle)
            t[i] = total_length

        # Normalize to [0, 1]
        t = t / t[-1]
        return t