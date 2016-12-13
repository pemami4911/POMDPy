class AlphaVector:
    """
    Simple wrapper for an alpha vector, used for representing the value function for a POMDP as a piecewise-linear,
    convex function
    """
    def __init__(self, a, v):
        self.action = a
        self.v = v

    def copy(self):
        return AlphaVector(self.action, self.v)
