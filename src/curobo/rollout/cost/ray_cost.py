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
        self.rays = torch.zeros((30, 3), device=torch.device("cuda:0"))
        CostBase.__init__(self, config)

    def look_at_obj_quaternion(self, camera_pos_batch, obj_center):
        #print("camera pos batch shape", camera_pos_batch.shape, obj_center.shape) #[batch, trajectory_points, 3], [1, 1, 3]
        # Compute "optimal" orientation (points toward object center)

#        obj_center = obj_center.to(camera_pos_batch.device)

        obj_center = self.obj_center
        obj_center = obj_center.expand_as(camera_pos_batch)

        # Direction vectors
        direction_vector_batch = obj_center - camera_pos_batch
        normalized_directions = F.normalize(direction_vector_batch, p=2, dim=-1)

        # Use standard reference vector
        ref_vec = self.ref_vec.expand_as(normalized_directions)

        # Compute rotation axes
        axes = torch.linalg.cross(ref_vec.expand_as(normalized_directions).float(), normalized_directions.float())
        normalized_axes = F.normalize(axes, p=2, dim=-1)

        # Compute rotation angles
        dot_products = torch.sum(normalized_directions.float() * ref_vec.expand_as(normalized_directions).float(), dim=-1)
        angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))

        # Construct quaternions
        w = torch.cos(angles / 2)
        xyz = torch.sin(angles / 2).unsqueeze(-1) * normalized_axes
        quaternions = torch.cat([w.unsqueeze(-1), xyz], dim=-1)

        return quaternions

    @staticmethod
    def direction_to_quaternion(direction):
        """
        Converts a unit direction vector into a quaternion that represents a rotation
        from the 'up' direction [0, 0, 1] to the given direction vector.
        
        Args:
            direction (torch.Tensor): A unit direction vector of shape [3], i.e., [vx, vy, vz].
            
        Returns:
            torch.Tensor: A quaternion of shape [4], representing the rotation.
        """
        # Ensure the input is a unit vector
        #if direction.norm() != 1.0:
        #    raise ValueError("Input direction must be a unit vector.")
        
        # Reference direction (the "up" vector)
        reference_direction = torch.tensor([0.0, 0.0, 1.0], device=direction.device).float()
        
        # Compute the axis of rotation: cross product between reference and direction
        axis_of_rotation = torch.cross(reference_direction.float(), direction.float(), dim=0)
        
        # Compute the angle between the vectors: dot product
        dot_product = torch.dot(reference_direction.float(), direction.float())
        angle = torch.acos(dot_product)
        
        # If the direction is already aligned with reference (i.e., dot_product is 1), no rotation is needed
        if torch.abs(dot_product - 1.0) < 1e-6:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], device=direction.device)  # No rotation
        
        # Normalize the axis of rotation
        axis_of_rotation = axis_of_rotation / axis_of_rotation.norm()
        
        # Create the quaternion
        half_angle = angle / 2.0
        sin_half_angle = torch.sin(half_angle)
        cos_half_angle = torch.cos(half_angle)
        
        quaternion = torch.cat([cos_half_angle.unsqueeze(0), axis_of_rotation * sin_half_angle])
        
        return quaternion

    def cost_scaling_very_front_heavy(self, cost, start=1000.0, cutoff_time=10):
        batch_size = cost.shape[0]
        size = cost.shape[1]
        end=0.0
        step = (end - start) / size
        out = torch.arange(start, end, step, device=torch.device("cuda:0"))
        out = out.unsqueeze(0).repeat(batch_size, 1)
        out[out < 0.0] = 0.0
        out[:, cutoff_time:] = 0.0
        return out

    def cost_scaling(self, cost, start=100.0, end=1000.0):
        batch_size = cost.shape[0]
        size = cost.shape[1]
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
        #print("cross product size", cross_product.shape)
        dist = torch.norm(cross_product, dim=-1) #batch, rays, trajectory
        #print("dist", dist.shape)
        #Should have solved the issue when there are no valid rays so rays size is 0, the min will throw an error for doing min on 0 size dim
        cost, min_indices = torch.min(dist, dim=1) #result batch, trajectory

        return cost #This is just the dist to the closest ray point

    #This one just randomly chooses one of the closest points
    @staticmethod
    def closest_point_on_the_ray_old(pose_camera_current, origin, rays): #[3], [3], [rays, 3]
        #calculate ray from origin to pose_camera_current
        pose_camera_current = pose_camera_current.unsqueeze(0)
        #print("pose_camera_current.shape", pose_camera_current.shape)
        origin = origin.unsqueeze(0)
        #print(origin.shape)
        v = pose_camera_current - origin #[1, 3]
        #print(v.shape)
        #v dot product with rays unit vectors
        t = torch.sum(v * rays, dim=-1).unsqueeze(-1) #[rays, 1]
        #print(t.shape)
        #Find these closest points on each ray
        #print((t * rays).shape)
        t = torch.clamp(t, min=0.0, max=0.9)
        points_closest = origin + (t * rays) #[rays, 3]
        #print(points_closest.shape)
        #Calculate distance from the pose_camera_current
        dist = torch.pow(pose_camera_current - points_closest, 2).sum(dim=-1) #[rays]
        #print(torch.pow(pose_camera_current - points_closest, 2).shape)
        #print(dist.shape)
        min_values, min_indices = torch.min(dist, dim=0)
        closest_point = points_closest[min_indices, :]
        min_indices = torch.randint(low=0, high=rays.shape[0]-1, size=(1,))
        closest_ray = rays[min_indices, :].squeeze()
        #closest_rotation = RayCost.direction_to_quaternion(closest_ray)
        print(closest_ray.shape)
        return closest_ray, None#closest_rotation

    @staticmethod
    def closest_point_on_the_ray(pose_camera_current, origin, rays): #[3], [3], [rays, 3]
        #calculate ray from origin to pose_camera_current
        pose_camera_current = pose_camera_current.unsqueeze(0)
        #print("pose_camera_current.shape", pose_camera_current.shape)
        origin = origin.unsqueeze(0)
        #print(origin.shape)
        v = pose_camera_current - origin #[1, 3]
        #print(v.shape)
        #v dot product with rays unit vectors
        t = torch.sum(v * rays, dim=-1).unsqueeze(-1) #[rays, 1]
        #print(t.shape)
        #Find these closest points on each ray
        #print((t * rays).shape)
        t = torch.clamp(t, min=0.1, max=0.5)
        points_closest = origin + (t * rays) #[rays, 3]
        #print(points_closest.shape)
        #Calculate distance from the pose_camera_current
        dist = torch.pow(pose_camera_current - points_closest, 2).sum(dim=-1) #[rays]
        #print(torch.pow(pose_camera_current - points_closest, 2).shape)
        #print(dist.shape)
        min_values, min_indices = torch.min(dist, dim=0)
        closest_point = points_closest[min_indices, :]
        closest_ray = rays[min_indices, :]
        closest_rotation = RayCost.direction_to_quaternion(closest_ray * -1.0)
        return closest_point, closest_rotation

    @staticmethod
    def closest_point_on_the_ray_dep(camera_pos_batch, origin, rays):
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
        #print("normalized current direction", normalized_current_direction)
        #print("camera_rot_batch", camera_rot_batch[0][0], camera_rot_batch[0][-1])
        #print("obj_center", obj_center[0][-1])
        # rospy.set_param("/start_ee_pos", camera_pos_batch[0, 0, :].tolist())
        # rospy.set_param("/cur_dir", normalized_current_direction[0, 0, :].tolist())
        # rospy.set_param("/des_dir", normalized_desired_direction[0, 0, :].tolist())

        #1 - desired vector dot product current vector 
        #cost = 1.0 - torch.dot(normalized_desired_direction, normalized_current_direction) #If they exactly match up, its 1
        dot_product = 1.0 - torch.sum(normalized_desired_direction * normalized_current_direction, dim=-1) #-1 to 1, so best is 0, worst is 2
        ori_cost = dot_product / 2.0
        ori_scale = self.cost_scaling_very_front_heavy(ori_cost, 20000, 20)
        ori_cost = ori_cost # * ori_scale
        #---------------------Orientation^

        #Position:
        #print("camera_pos_batch.shape", camera_pos_batch.shape)
        # trajectory_divider = max(camera_pos_batch.shape[1] // 2, 1)
        # #print("trajectory divider", trajectory_divider)
        # front_part_camera_post_batch = camera_pos_batch[:, 0:trajectory_divider, :]
        # pos_cost = self.closest_ray_cost(front_part_camera_post_batch, origin, rays) #(batch, trajectory)
        # #slack = torch.zeros(size=[camera_pos_batch.shape[0], camera_pos_batch.shape[1] - trajectory_divider]).to('cuda')
        # slack = camera_pos_batch[:, :, 0] * 0.0 #(batch, trajectory) all zeros
        # #print("slack shape", slack.shape)
        # slack = slack[:, trajectory_divider:]
        # #print("slack shape 2", slack.shape)
        # pos_cost = torch.cat([pos_cost, slack], dim=1) #(batch, trajectory/2) cat (batch, trajectory/2)
        pos_cost = self.closest_ray_cost(camera_pos_batch, origin, rays)
        pos_scale = self.cost_scaling_very_front_heavy(pos_cost, 2000, 20)
        pos_cost = pos_cost # * pos_scale
        #

        """
        Notes:
        Flat scaling makes it much smoother than earlier scaling
        I see an issue of moving towards only the nearest ray which doesn't even give the best view information.
        I also see that the centerpoint of the rays do not give the best place to actually look in terms of efficiency to look at occluded space from the current position.
        Those rays might make sense as full goals, but small adjustments on the trajectory are not helped by those rays at all. 
        Lets try for small adjustments for occluded volume closest to the end effector. 
        If all rays are in collision I do backwards rays but the cost to be close to the rays is high and that stops the progress entirely as it tries to get to the z.

        Ok so there is something to say about the rays being too far so not helpful on the way to the grasp and wanting something closer for small optimizations
        It seems that when we go for a grasp we will already have looked at the easy nearby rays and the ray cost will be more of just a hindrance on the grasping trajectory
        There is something to say about the closest ray maybe not being the most useful ray to go to while there is a large viewpoint of many good rays elsewhere
        I am also seeing not full commital where it gets closer which doesn't actually do anything, so it needs to be full commital as much higher reward or something. 

        I ran many experiments with the new cost function and without an auxiliary cost function:
        The costs are working correctly, I tried just position and that was clearly going to the closest ray and I tried just orientation and that worked too.
        I have the max of pos and rot scaled by a higher number and the min of pos and rot scaled by a lower number that is /10 of the higher number
        Flat scaling is much less jerky.
        There is a bit of randomness with the occlusion volume and that changes the trajectory taken across identical runs. 
        Here are some pictures of the resulting occlusion volume after getting to the grasp pose with no cost and with auxiliary cost.
        The auxiliary cost is consistently getting better visibility.
        """

        #Now put pos_cost between 0 and 1 where it is the distance, so pos_cost / max_distance where I will say like 5 is good
        max_dist = 3.0
        #final_cost = (ori_cost * 5000) + (pos_cost * 5000 / max_dist)
        flat = True
        large_scale = 20000
        small_scale = 2000
        if flat == False:
            large_scale = self.cost_scaling(ori_cost, start=1000.0, end=20000.0)
            small_scale = self.cost_scaling(ori_cost, start=100.0, end=2000.0)
        important_cost = large_scale * torch.maximum(ori_cost, pos_cost / max_dist) #[batch, trajectory], [batch, trajectory]
        less_important_cost = small_scale * torch.minimum(ori_cost, pos_cost / max_dist)
        final_cost = important_cost + less_important_cost

        return final_cost.float() * 0.7

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