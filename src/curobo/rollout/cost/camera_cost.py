from curobo.rollout.cost.pose_cost import *
import torch.nn.functional as F

# TODO: test this implementation
# TODO: fix inheritance (PoseCost is not used at all right now. Is a config class even necessary right now?)
class CameraCost(CostBase):
    def __init__(self, config: CostConfig = None):
        self.weight = torch.tensor(1, device=torch.device("cuda:0"))
        self.tensor_args = TensorDeviceType()
        self.obj_center = torch.tensor([[[1.05197, -.219925, 1.03373]]], device=torch.device("cuda:0")) #Mustard 02 perishables
        #self.obj_center = torch.tensor([[[1.45, -.15, 1.22]]], device=torch.device("cuda:0")) 
        #self.obj_center = torch.tensor([1.12, 0.140003, 1.05591], device=torch.device("cuda:0")) #Chips can
        self.ref_vec = torch.tensor([1, 0, 0], device=torch.device("cuda:0"))
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


    # Handles multiple configuration, single goal
    # Right now the end effector wants to look at it, and not necessarily the camera
    # Want the camera to point at the object, so just get the camera rotational pose and evaluate that based on how well it is facing the desired position
    # def forward(self, camera_pos_batch, camera_rot_batch, obj_center, link_name : str = None):
    #     goal_camera_rot_batch = self.look_at_obj_quaternion(camera_pos_batch, obj_center)

    #     dot_products = torch.sum(goal_camera_rot_batch * camera_rot_batch, dim=-1)

    #     # Clamped to prevent potential errors due to numerical precision
    #     dot_products = torch.clamp(dot_products, -1.0, 1.0)

    #     #1 = 0, -1 = 3.14
    #     distances = torch.acos(dot_products) #don't do abs as it needs to know the sign, chatgpt gets this wrong here

    #     # print("COMPUTING CAMERA COST")
        
    #     #print("distances scaled", 10.0 * distances) #average is around 5   

    #     #[batch, trajectory, ee xyz]
    #     #return -10000 + camera_pos_batch[:, :, -1] * 10000.0 #The higher z is, the lower the cost
    #     #At the worst its 3.14e5, make its best as 0
    #     return 20000.0 * distances

    # def forward(self, camera_pos_batch, camera_rot_batch, obj_center, link_name: str = None):
    #     #print("camera pos batch size inp", camera_pos_batch.shape)
    #     # Compute the desired quaternion

    #     #1 - desired vector dot product current vector

    #     goal_camera_rot_batch = self.look_at_obj_quaternion(camera_pos_batch, obj_center)

    #     # Compute dot products between quaternions
    #     dot_products = torch.sum(goal_camera_rot_batch * camera_rot_batch, dim=-1)

    #     # Enforce the shortest rotation by taking the absolute value of dot products
    #     dot_products = torch.abs(dot_products)

    #     # Clamp dot products to avoid numerical issues with acos
    #     dot_products = torch.clamp(dot_products, -1.0, 1.0)

    #     # Compute angular distances (in radians)
    #     distances = 2.0 * torch.acos(dot_products)  # Multiply by 2 to get the full rotation angle

    #     # Scale the loss
    #     return 1000.0 * torch.exp(distances) - 1000.0

    def forward(self, camera_pos_batch, camera_rot_batch, obj_center, link_name: str = None):
        #print("camera pos batch size inp", camera_pos_batch.shape)
        # Compute the desired quaternion

        obj_center = self.obj_center
        obj_center = obj_center.expand_as(camera_pos_batch)

        #Desired direction vectors
        direction_vector_batch = obj_center - camera_pos_batch
        normalized_desired_direction = F.normalize(direction_vector_batch, p=2, dim=-1)

        #Current vector
        normalized_current_direction = quaternion_to_direction(camera_rot_batch)

        #1 - desired vector dot product current vector 
        #cost = 1.0 - torch.dot(normalized_desired_direction, normalized_current_direction) #If they exactly match up, its 1
        dot_product = 1.0 - torch.sum(normalized_desired_direction * normalized_current_direction, dim=-1) #-1 to 1, so best is 0, worst is 2
        cost = dot_product

        batch_size = cost.shape[0]
        size = cost.shape[1]
        start=1.0
        end=10000.0
        step = (end - start) / size
        out = torch.arange(start, end, step, device=torch.device("cuda:0"))
        out = out.unsqueeze(0).repeat(batch_size, 1)
        #print(out.shape, cost.shape)

        return cost * out

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