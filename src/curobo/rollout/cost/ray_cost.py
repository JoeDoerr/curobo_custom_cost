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
        self.weight = torch.tensor(1.0, device=torch.device("cuda:0"))
        self.tensor_args = TensorDeviceType()
        self.ref_vec = torch.tensor([1, 0, 0], device=torch.device("cuda:0"))
        self.origin = torch.zeros((1, 3), device=torch.device("cuda:0"))
        self.rays = torch.zeros((1, 3), device=torch.device("cuda:0"))
        self.collision_free_rays = torch.zeros((72, 3), device=torch.device("cuda:0"))
        self.needed_steps = torch.tensor([0], device=torch.device("cuda:0"))

        num_origins=1
        self.cylinders = torch.zeros((num_origins, 72, 3), device=torch.device("cuda:0"), requires_grad=False)
        self.radii = torch.zeros((num_origins, 72), device=torch.device("cuda:0"))
        self.origins = torch.zeros((num_origins, 3), device=torch.device("cuda:0"), requires_grad=False)
        self.sigmoid_steepness = torch.tensor([350.0], device=torch.device("cuda:0"))
        self.attractor_decay = torch.tensor([1.0], device=torch.device("cuda:0"))

        self.always_look_here = torch.zeros((3), device=torch.device("cuda:0"))
        CostBase.__init__(self, config)

   
       #This one does the discrete-aware visibility costs
    def forward(self, camera_pos_batch, camera_rot_batch): #rays = [rays, 3], origin=[1, 3]
        """
        Essentially, the ori and pos distances are added up and the minimum together on the ray dimension are chosen, then the minimum on the point dimension
        Already seen points are not eligible for minimum which essentially removes them

        The distance from the current position and rotation is checked against every single ray
        For each state in the each trajectory, the threshold for if a ray is seen or not is calculated. 
        The already seen points have all their rays set to 7 so they don't get considered in choosing the lowest value
        Then the min of the rays are chosen per point
        Then the already seen points are set to all 0 so the cost goes down
        Then the output is the nearest ray to every configuration that isn't a ray that has been seen yet

        It wants the good rotation so it stay on the rotation then moves, then stays again, doing these odd jumps. 
        It doesn't go to the position and just goes halfway because it isn't worth it to go there.
        It wants the positional and rotational costs to go down as soon as possible averaged along the trajectory.
        But why isn't it totally going there in the meantime and instead only doing part-way? Maybe the line search?
        """
        #For each point and for each of those point's rays, (broadcast with [1, 1] in the front) check the straight line distance and the rotation distance
        #So calculate distances as normal but keep the [batch, trajectory, points, rays, 1] for pos and rot
        #Then min lowest of 3 on the rays
        #Then modify if with ranges

        origins = self.origins
        rays = self.cylinders

        #---------------------Orientation:
        obj_center = origins.unsqueeze(0).unsqueeze(0) #[1, 1, points, 3] for [batch, trajectory, points, 3]
        # #print("obj", obj_center.shape)

        # #Desired direction vectors
        direction_vector_batch = obj_center - camera_pos_batch.unsqueeze(-2) #[1, 1, points, 3] - [batch, trajectory, 1, 3]
        # #print("dir", direction_vector_batch.shape)
        normalized_desired_direction = F.normalize(direction_vector_batch, p=2, dim=-1).unsqueeze(-2) #[batch, trajectory, points, 1(rays), 3]
        #normalized_desired_direction = -1.0 * rays.unsqueeze(0).unsqueeze(0) #[1, 1, points, rays, 3]

        #Current vector
        normalized_current_direction = quaternion_to_direction(camera_rot_batch)
        normalized_current_direction = normalized_current_direction.unsqueeze(-2).unsqueeze(-2) #[batch, trajectory, 1, 1, 3]
        #print("desired orientation shapes", normalized_desired_direction.shape, normalized_current_direction.shape)

        #1 - desired vector dot product current vector
        #cost = 1.0 - torch.dot(normalized_desired_direction, normalized_current_direction) #If they exactly match up, its 1
        dot_product = 1.0 - torch.sum(normalized_desired_direction * normalized_current_direction, dim=-1) #-1 to 1, so best is 0, worst is 2
        ori_distances = dot_product / 2.0 #best is 0, worst is 2 [batch, trajectory, points, 1]

        #---------------------Orientation for always look here:
        obj_center2 = self.always_look_here.unsqueeze(0).unsqueeze(0) #[1, 1, 3]

        # #Desired direction vectors
        direction_vector_batch2 = obj_center2 - camera_pos_batch #[1, 1, 3] - [batch, trajectory, 3]
        normalized_desired_direction2 = F.normalize(direction_vector_batch2, p=2, dim=-1) #[batch, trajectory, 3]

        #Current vector
        normalized_current_direction2 = quaternion_to_direction(camera_rot_batch) #[batch, trajectory, 3]
        #print("desired orientation shapes", normalized_desired_direction.shape, normalized_current_direction.shape)

        #1 - desired vector dot product current vector
        dot_product2 = 1.0 - torch.sum(normalized_desired_direction2 * normalized_current_direction2, dim=-1) #-1 to 1, so best is 0, worst is 2
        ori_distances_always_look_here = dot_product2 / 2.0 #[batch, trajectory]

        #---------------------Position
        pos_distances = self.ray_pos_distance(camera_pos_batch.unsqueeze(-2), origins.unsqueeze(0).unsqueeze(0), rays) #[batch, trajectory, points, rays]
        
        total_costs = ori_distances + pos_distances

        final_cost_points, _ = torch.min(total_costs, dim=-1) #[batch, trajectory, points, rays] result = [batch, trajectory, points]
        final_cost, _ = torch.min(final_cost_points, dim=-1) #[batch, trajectory, points] result = [batch, trajectory]

        scaler = 10000.0
        mask = final_cost > 1e10
        center_idx = final_cost.shape[1] // 2
        indices = torch.full((final_cost.shape[0], final_cost.shape[1]), center_idx, dtype=torch.long, device=torch.device("cuda:0"))
        mask = mask.scatter_(1, indices, True)
        final_cost = final_cost * mask
        final_cost = final_cost * scaler

        ori_distances_always_look_here = ori_distances_always_look_here * ~mask
        ori_distances_always_look_here = ori_distances_always_look_here * (scaler / 50.0)
        final_cost = final_cost + ori_distances_always_look_here

        return final_cost.float() #* self.weight
    
    #origin to camera pose vector cross product each ray then we only want to be attracted to the closest ray so we do the norm then min
    def ray_pos_distance(self, camera_pos_batch, origins, rays): #[batch, trajectory, 1, 3], [1, 1, points, 3], [1, points, rays, 3]
        #Calculate the vector of origin to end effector pos
        #print("cam", camera_pos_batch.shape, origins.shape)
        v = camera_pos_batch - origins #(batch, trajectory, points, 3) This is origin to camera pos batch as if i do + origin it will point me to camera_pos_batch
        #v is [batch, trajectory, points, 3]

        #Cross product each ray against each v (batch, trajectory, points, 1, 3) cross (1, 1, points, rays, 3) = (batch, trajectory, points, rays, 3)
        #print("camera", v.shape, rays.shape, v.unsqueeze(-2).shape, rays.unsqueeze(0).shape)
        cross_product = torch.cross(v.unsqueeze(-2), rays.unsqueeze(0).unsqueeze(0), dim=-1) #[batch, trajectory, points, 1, 3) and (1, 1, points, rays, 3)
        #print("cross product size", cross_product.shape)
        dist = torch.norm(cross_product, dim=-1) #[batch, trajectory, points, rays]
        return dist

    #This one does the discrete-aware visibility costs
    def forward_old2(self, camera_pos_batch, camera_rot_batch): #rays = [rays, 3], origin=[1, 3]
        """
        Essentially, the ori and pos distances are added up and the minimum together on the ray dimension are chosen, then the minimum on the point dimension
        Already seen points are not eligible for minimum which essentially removes them

        The distance from the current position and rotation is checked against every single ray
        For each state in the each trajectory, the threshold for if a ray is seen or not is calculated. 
        The already seen points have all their rays set to 7 so they don't get considered in choosing the lowest value
        Then the min of the rays are chosen per point
        Then the already seen points are set to all 0 so the cost goes down
        Then the output is the nearest ray to every configuration that isn't a ray that has been seen yet

        It wants the good rotation so it stay on the rotation then moves, then stays again, doing these odd jumps. 
        It doesn't go to the position and just goes halfway because it isn't worth it to go there.
        It wants the positional and rotational costs to go down as soon as possible averaged along the trajectory.
        But why isn't it totally going there in the meantime and instead only doing part-way? Maybe the line search?
        """
        #For each point and for each of those point's rays, (broadcast with [1, 1] in the front) check the straight line distance and the rotation distance
        #So calculate distances as normal but keep the [batch, trajectory, points, rays, 1] for pos and rot
        #Then min lowest of 3 on the rays
        #Then modify if with ranges

        origins = self.origins
        rays = self.cylinders

        #---------------------Orientation:
        obj_center = origins.unsqueeze(0).unsqueeze(0) #[1, 1, points, 3] for [batch, trajectory, points, 3]
        # #print("obj", obj_center.shape)

        # #Desired direction vectors
        direction_vector_batch = obj_center - camera_pos_batch.unsqueeze(-2) #[1, 1, points, 3] - [batch, trajectory, 1, 3]
        # #print("dir", direction_vector_batch.shape)
        normalized_desired_direction = F.normalize(direction_vector_batch, p=2, dim=-1).unsqueeze(-2) #[batch, trajectory, points, 1(rays), 3]
        #normalized_desired_direction = -1.0 * rays.unsqueeze(0).unsqueeze(0) #[1, 1, points, rays, 3]

        #Current vector
        normalized_current_direction = quaternion_to_direction(camera_rot_batch)
        normalized_current_direction = normalized_current_direction.unsqueeze(-2).unsqueeze(-2) #[batch, trajectory, 1, 1, 3]
        #print("desired orientation shapes", normalized_desired_direction.shape, normalized_current_direction.shape)

        #1 - desired vector dot product current vector
        #cost = 1.0 - torch.dot(normalized_desired_direction, normalized_current_direction) #If they exactly match up, its 1
        dot_product = 1.0 - torch.sum(normalized_desired_direction * normalized_current_direction, dim=-1) #-1 to 1, so best is 0, worst is 2
        ori_distances = dot_product / 2.0 #best is 0, worst is 2 [batch, trajectory, points, 1]
        #ori_distances *= 0.0
        """
        10 Degrees is 0.984 for the dot product where then (1-0.984) / 2 = 0.00759
        350 steepness sigmoid is from the middle to the outsides is 0.018 which means leeway = 0.00759 + 0.018 too much
        1000 steepness sigmoid is 0.003 which makes leeway = 0.00759 + 0.003 which is something around 15 degrees
        """
        original_ori_distances = ori_distances
        max_ori_distance = 0.00759
        #print("ori distances1", ori_distances[0, 0], ori_distances.shape)
        ori_distances_check = ori_distances - max_ori_distance #Need to make sure this is not -= as that will be an in place modification to the pointer of ori_distances which original is set to
        ori_distances_check = torch.sigmoid(ori_distances_check * 1000.0) #[batch, trajectory, points, rays]
        #print("ori", ori_distances.shape)

        #---------------------Position
        pos_distances = self.ray_pos_distance(camera_pos_batch.unsqueeze(-2), origins.unsqueeze(0).unsqueeze(0), rays) #[batch, trajectory, points, rays]
        #pos_distances = pos_distances
        original_pos_distances = pos_distances
        #pos_distances, min_indices = torch.min(pos_distances, dim=-1) #[batch, trajectory, points], min_indices is just the reduced dim min is over
        max_pos_distance = self.radii.unsqueeze(0).unsqueeze(0).repeat(pos_distances.shape[0], pos_distances.shape[1], 1, 1) #[points, rays] -> [batch, trajectory, points, rays]
        #max_pos_distance = torch.gather(max_pos_distance, dim=-1, index=min_indices.unsqueeze(-1)).squeeze(-1) #[batch, trajectory, points]
        #print("Max pos distance", max_pos_distance.shape)
        #print("dists", pos_distances.shape, max_pos_distance.shape, self.radii.shape)
        #print("pos distances1", pos_distances[0, 0], max_pos_distance[0, 0])
        """
        The radius as a max distance is fine for the maximum distance with a tiny bit
        The minimum radius is 0.02 for the steepness, where sigmoid steepness 500 is 0.005 is one fourth the leeway
        """
        pos_distances_check = pos_distances - (max_pos_distance * 0.9) #Need to make sure this is not -= as that will be an in place modification to the pointer of pos_distances which original is set to
        pos_distances_check = torch.sigmoid(pos_distances_check * 500.0) #Changing the sigmoid steepness will make the ori and pos easier to get gradients
        # lowest_val, _ = torch.min(pos_distances, dim=-1)
        # lowest_val, _ = torch.min(lowest_val, dim=-1)
        # lowest_val2, _ = torch.min(ori_distances, dim=-1)
        # lowest_val2, _ = torch.min(lowest_val2, dim=-1)
        # print("pos", lowest_val[0, :], "ori", lowest_val2[0, :]) #Across points on a trajectory

        #---------------------Mask out values where not both pos and rotation fit
        # total_distance_mask = torch.max(ori_distances, pos_distances) < 0.99 #[batch, trajectory, points] where all that is false will be set to 0 which is when no rate of change
        ori_mean = torch.abs(ori_distances.mean()).detach() + 1e-10
        pos_mean = torch.abs(pos_distances.mean()).detach() + 1e-10
        #print("ori mean, pos mean", ori_mean, pos_mean)
        # total_costs = (ori_distances / ori_mean) + (pos_distances / pos_mean)
        total_costs = ori_distances + pos_distances
        # total_costs = total_costs * total_distance_mask #IMPORTANT: The only pos and ori pairs that have change is when both are lower than 0.99 output from sigmoid
        # high_cost_mask = total_costs != 0.0
        # total_costs = torch.where(high_cost_mask, total_costs, 1.0)

        #print("total distances", total_costs[0, 0], pos_distances[0, 0])
        #stepped_costs = torch.sigmoid(total_distances * self.sigmoid_steepness) #[batch, trajectory, points]
        #print("stepepsed", stepped_costs.shape)
        """
        Each trajectory point in [trajectory, 50] has 50 values. I want to make a truth value for each of the 50 values for if it is satisfying a condition. 
        Then I want a mask that says true only on the first instance of this index of 50 being true. Everything that doesn't match the condition is also true
        """
        #print("stepped costs", stepped_costs[0, 0])

        #---------------------Only count points once in the trajectory
        #For each point, find if there is a joined distance for all of its rays for that point close enough to have been seen
        min_pos_dist, _ = torch.min(pos_distances_check, dim=-1) #[batch, trajectory, points]
        min_ori_dist, _ = torch.min(ori_distances_check, dim=-1) #[batch, trajectory, points]
        condition = torch.max(min_ori_dist, min_pos_dist) <= 0.5 #[batch, trajectory, points] where 0.5 means at the edge of max dist, so both in threshold
        #First instance along the trajectory dimension
        #Down the dimension it accumulates the values but doesn't collapse the lower dimensions
        #So cumsum adds a [point] tensor for the first trajectory point, then keeps going and adds the next [50] tensors to it
        #All of the one values are the first values. There are intermediate 1 values in the accumulation so just do & with the original condition
        first_true_mask = condition & (condition.cumsum(dim=1) == 1) #This is the first time it was true
        #print("fist", first_true_mask.shape)
        final_mask = torch.logical_or(first_true_mask, ~condition) #[batch, trajectory, points] ~condition is everywhere its not true so keep gradients there
        #print("final mask", final_mask.shape)
        final_mask_high = final_mask.unsqueeze(-1).repeat(1, 1, 1, total_costs.shape[-1])
        #total_costs = torch.where(final_mask_high, total_costs, 7.0) #Each seen point's entire ray set on the trajectory that is seen is set high temporarily

        #---------------------Removing meant-to-be removed too small radius rays
        # remove_mask = total_distances < 90 #True keep
        # total_costs = torch.where(remove_mask, total_costs, 0) #Where its false is set to 0

        #---------------------Smooth attractor addition
        # smooth_attractors = original_ori_distances + original_pos_distances
        # smooth_attractors = torch.where(final_mask, smooth_attractors, 0) #Temporal removing from sigmoid knowledge #[batch, trajectory, points, rays]
        # smooth_attractors, _ = torch.min(smooth_attractors, dim=-1) #lowest distance ray
        # smooth_attractor_sum = torch.sum(smooth_attractors, dim=-1) #[batch, trajectory]
        # smooth_attractor_sum = smooth_attractor_sum * 0.0 #self.attractor_decay
        #self.attractor_decay *= 0.9

        final_cost_points, _ = torch.min(total_costs, dim=-1) #[batch, trajectory, points, rays] result = [batch, trajectory, points]
        final_cost, _ = torch.min(final_cost_points, dim=-1) #[batch, trajectory, points] result = [batch, trajectory]
        #final_cost = torch.where(final_mask, final_cost, 0.0) #Setting already seen to 0 so the overall cost goes down
        #print("final cost", final_cost[0, :])
        #final_cost = torch.sum(final_cost, dim=-1) #[batch, trajectory] #This sum should have large difference with 1 seen and less with more seen
        #final_cost += smooth_attractor_sum
        #print("fincos", final_cost.shape)
        final_cost *= 30.0
        reduce_gradients_and_value_when_seen_more = torch.where(final_mask, 100.0, 1.0).mean() #[batch, trajectory, points] The more false the lower to scale, the less false the higher to scale
        final_cost *= reduce_gradients_and_value_when_seen_more
        #print("reduce_gradients_and_value_when_seen_more", reduce_gradients_and_value_when_seen_more)

        #print("weight", self.weight)

        return final_cost.float() #* self.weight

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
    
    def direction_to_quaternion_multiple_randomized(directions: torch.Tensor) -> torch.Tensor:
        """
        Convert unit direction vectors (N, 3) into quaternions (N, 4),
        by randomly assigning the missing rotational degree of freedom.
        
        :param directions: Tensor of shape (N, 3) representing unit direction vectors.
        :return: Tensor of shape (N, 4) representing quaternions.
        """
        # Ensure the input is a unit vector
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # Generate random angles for the missing degree of freedom
        random_angles = torch.rand(directions.shape[0]) * 2 * torch.pi
        
        # Choose an arbitrary reference vector different from the input direction
        reference = torch.tensor([0.0, 0.0, 1.0], device=directions.device).expand_as(directions)
        mask = torch.abs(torch.sum(directions * reference, dim=-1)) > 0.99
        reference[mask] = torch.tensor([1.0, 0.0, 0.0], device=directions.device)
        
        # Compute perpendicular vector to form a frame
        right = torch.cross(reference, directions)
        right = right / torch.norm(right, dim=-1, keepdim=True)
        
        # Compute the second perpendicular vector
        up = torch.cross(directions, right)
        
        # Construct rotation quaternions
        cos_half_angle = torch.cos(random_angles / 2).unsqueeze(-1)
        sin_half_angle = torch.sin(random_angles / 2).unsqueeze(-1)
        
        x = right[:, 0] * sin_half_angle + up[:, 0] * sin_half_angle
        y = right[:, 1] * sin_half_angle + up[:, 1] * sin_half_angle
        z = right[:, 2] * sin_half_angle + up[:, 2] * sin_half_angle
        w = cos_half_angle.squeeze(-1)
        
        return torch.stack([w, x, y, z], dim=-1)

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
    def closest_points_on_the_ray(pose_camera_current, origin, rays): #[3], [3], [rays, 3]
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
        #t = torch.clamp(t, min=0.1, max=0.5)
        points_closest = origin + (t * rays) #[rays, 3]
        closest_rotations = RayCost.direction_to_quaternion_multiple_randomized(points_closest * -1.0)
        return points_closest, closest_rotations

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

    def forward_older(self, camera_pos_batch, camera_rot_batch, origin = None, rays = None): #rays = [rays, 3], origin=[1, 3]
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
        #ori_scale = self.cost_scaling_very_front_heavy(ori_cost, 20000, 20)
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
        pos_cost_best_ray = self.closest_ray_cost(camera_pos_batch, origin, rays)
        pos_cost_all_rays = self.closest_ray_cost(camera_pos_batch, origin, self.collision_free_rays) #All the rays
        #pos_scale = self.cost_scaling_very_front_heavy(pos_cost, 2000, 20)
        #pos_cost_best_ray[:, self.needed_steps-1] *= 1.0
        #print("self.needed_steps-1", self.needed_steps-1)
        pos_cost_best_ray[:, :-1] *= 0.01
        pos_cost_all_rays[:, -1] *= 0.5
        pos_cost = pos_cost_best_ray + pos_cost_all_rays

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
        max_dist = 2.0
        use_different_scales=False
        if use_different_scales == True:
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
        else:
            pos_cost = pos_cost / max_dist
            final_cost = ori_cost + pos_cost
            final_cost *= 2000 #20000
        
        #final_cost = ori_cost
        #final_cost *= 200

        return final_cost.float() * self.weight

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