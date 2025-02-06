from curobo.rollout.cost.pose_cost import *

"""
Take in a set of grasps
Return a cost based on the distances to each of the grasps

Works in the following cases
- No grasps
- Unreliable grasps
- Reliable grasps
"""

class MultiGraspCost(PoseCost):
    def forward(self, **kwargs):
        return super()(**kwargs)