from curobo.rollout.cost.pose_cost import *
import torch.nn.functional as F
import rospy
import time
import torch

"""
Take in the set of non-colliding rays
Move towards the closet n rays to the given position or choose one out a range of closest randomly so that we don't move to an averaged position
Heavily front-weight this as once we get to the closet ray it will no longer be a ray (do high start to negative end, and then do all negative turn to 0 for the scaling)
"""

class RayCost(CostBase):
    def __init__(self, config: CostConfig = None):
        self.weight = torch.tensor(1, device=torch.device("cuda:0"))
        self.tensor_args = TensorDeviceType()
        self.ref_vec = torch.tensor([1, 0, 0], device=torch.device("cuda:0"))
        self.origin = torch.zeros((1, 3), device=torch.device("cuda:0"))
        self.rays = torch.zeros((36, 3), device=torch.device("cuda:0"))
        CostBase.__init__(self, config)

    def cost_scaling_very_front_heavy(self, cost):
        batch_size = cost.shape[0]
        size = cost.shape[1]
        start=1000.0
        end=-1000.0
        step = (end - start) / size
        out = torch.arange(start, end, step, device=torch.device("cuda:0"))
        out = out.unsqueeze(0).repeat(batch_size, 1)
        out[out < 0.0] = 0.0
        return out

    #origin to camera pose vector cross product each ray then we only want to be attracted to the closest ray so we do the norm then min
    def closest_ray_cost(self, camera_pos_batch, origin, rays):
        #Calculate the vector of origin to end effector pos
        v = camera_pos_batch - origin.expand_as(camera_pos_batch) #(batch, trajectory, 3) This is origin to camera pos batch as if i do + origin it will point me to camera_pos_batch
        #Dot product of rays with v to get the amount that they line up
        e_rays = rays.unsqueeze(1).unsqueeze(0) #(1, rays, 1, 3)
        #e_camera_pos_batch = camera_pos_batch.unsqueeze(1) #(batch, 1, trajectory, 3)
        #Cross product each ray against each v (batch, 1, trajectory, 3) cross (1, rays, 1, 3) = (batch, rays, trajectory, 3)
        cross_product = torch.cross(v.unsqueeze(1), e_rays, dim=-1)
        dist = torch.norm(cross_product, dim=-1) #batch, rays, trajectory
        cost, min_indices = torch.min(dist, dim=1) #batch, trajectory

        return cost #This is just the dist to the closest ray point

    def closest_point_on_the_ray_not_used(self, camera_pos_batch, origin, rays):
        #Calculate the vector of origin to end effector pos
        v = camera_pos_batch - origin.expand_as(camera_pos_batch) #(batch, trajectory, 3) This is origin to camera pos batch as if i do + origin it will point me to camera_pos_batch
        #Dot product of rays with v to get the amount that they line up
        e_rays = rays.unsqueeze(1).unsqueeze(0) #(1, rays, 1, 3)
        e_camera_pos_batch = camera_pos_batch.unsqueeze(1) #(batch, 1, trajectory, 3)
        t_int = e_camera_pos_batch * e_rays #(batch, rays, trajectory, 3)
        t = t_int.sum(dim=-1) #(batch, rays, trajectory) Now dot product is complete
        points_closest = origin.expand_as(t_int) + (e_rays * t.unsqueeze(-1)) #(1, 1, 1, 3) + (1, rays, 1, 3) * (batch, rays, trajectory, 1) = (batch, rays, trajectory, 3)
        #Then from the closest ray calculate euclidean distance as cost
        dist = torch.pow((camera_pos_batch.unsqueeze(1) - points_closest), 2).sum(dim=-1) #(batch, rays, trajectory)
        #Min to find the closest ray and its distance is the cost, will differentiate so other rays simply won't be seen
        min_values, min_indices = torch.min(dist, dim=1) #min the ray dimension so now each batch and each trajectory point have a distance value to closest ray to result in (batch, trajectory)
        #Now min_indicies indicates what ray out of [batch, rays, trajectory, 3] should be chosen to be [batch, trajectory, 3]
        #Then we calculate the cross product of t = (batch, )

        return min_values #This is just the dist to the closest ray point

    def forward(self, camera_pos_batch, camera_rot_batch, origin = None, rays = None): #rays = [rays, 3], origin=[1, 3]
        origin = self.origin
        rays = self.rays

        #---------------------Orientation:
        obj_center = origin.expand_as(camera_pos_batch)

        #Desired direction vectors
        direction_vector_batch = obj_center - camera_pos_batch
        normalized_desired_direction = F.normalize(direction_vector_batch, p=2, dim=-1)

        #Current vector
        normalized_current_direction = quaternion_to_direction(camera_rot_batch)

        #1 - desired vector dot product current vector 
        #cost = 1.0 - torch.dot(normalized_desired_direction, normalized_current_direction) #If they exactly match up, its 1
        dot_product = 1.0 - torch.sum(normalized_desired_direction * normalized_current_direction, dim=-1) #-1 to 1, so best is 0, worst is 2
        ori_cost = dot_product

        #---------------------Orientation^

        #Position:
        pos_cost = self.closest_ray_cost(camera_pos_batch, origin, rays)
        #

        cost = ori_cost + pos_cost

        scale = self.cost_scaling_very_front_heavy(cost)

        return cost * scale

def quaternion_to_direction(quaternions):
    """
    Convert a batch of quaternions to normalized direction vectors.
    
    Args:
        quaternions (torch.Tensor): A tensor of shape [batch, tensor, 4] representing quaternions 
                                    in (w, x, y, z) format.

    Returns:
        torch.Tensor: A tensor of shape [batch, tensor, 3] representing the direction vectors.
    """
    # Ensure quaternions are normalized
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    # Extract components
    w, x, y, z = quaternions.unbind(dim=-1)
    
    # Compute the forward direction (unit vector along the z-axis in local space rotated by the quaternion)
    forward_x = 2 * (x * z + w * y)
    forward_y = 2 * (y * z - w * x)
    forward_z = 1 - 2 * (x**2 + y**2)
    
    # Combine into direction vectors
    directions = torch.stack([forward_x, forward_y, forward_z], dim=-1)
    
    # Normalize the direction vectors to ensure unit length
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    return directions