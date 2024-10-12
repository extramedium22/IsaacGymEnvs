# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

class DroneAgent(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 10000
        self.max_iterations = 500000

        #设定观测维度与控制维度
        self.cfg["env"]["numObservations"] = 10
        self.cfg["env"]["numActions"] = 4

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        #这里获取仿真环境的状态
        '''
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[:, :, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        print("第一个环境的pos:" + str(self.dof_pos[0,...]))
        print("所有环境的pos:" + str(self.dof_pos))
        print("DOF的reshape:" + str(self.dof_state.view(self.num_envs, self.num_dof, 2)))
        '''
        
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        #root_state_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)
        root_state_tensor = self.root_tensor.view(self.num_envs, 13)
        self.root_states = root_state_tensor
        self.root_positions = self.root_states[..., 0:3]
        self.root_linvels = self.root_states[..., 7:10]
        self.root_quaternions = self.root_states[..., 3:7]
        #self.root_angles = get_euler_xyz(self.root_quaternions)
        #self.root_angles_phi, self.root_angles_theta, self.root_angles_psi = get_euler_xyz(self.root_quaternions)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_root_states = root_state_tensor.clone()


        #self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        #vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)
        #self.root_states = vec_root_tensor
        #self.root_positions = vec_root_tensor[..., 0:3]
        #self.root_linvels = vec_root_tensor[..., 7:10]
        #self.gym.refresh_actor_root_state_tensor(self.sim)
        #self.initial_root_states = vec_root_tensor.clone()

        
        

    def create_sim(self):
        #print("开始创建sim")
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        #print("创建sim完成")

    def _create_ground_plane(self):
        #print("开始生成地面")
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)
        #print("生成地面完成")

    def _create_envs(self, num_envs, spacing, num_per_row):
        #print("开始创建环境")
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/drone.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        drone_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        #self.num_dof = self.gym.get_asset_dof_count(drone_asset)
        #self.num_dof = 3
        #print("总自由度num_dof=" + str(self.num_dof))

        pose = gymapi.Transform()
        '''
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)
        '''
        pose.p.z = 2
        pose.r = gymapi.Quat(1.0, 0.0, 0.0, 0.0)


        self.drone_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            drone_handle = self.gym.create_actor(env_ptr, drone_asset, pose, "drone", i, 1, 0)

            '''
            dof_props = self.gym.get_actor_dof_properties(env_ptr, drone_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, drone_handle, dof_props)
            '''

            self.envs.append(env_ptr)
            self.drone_handles.append(drone_handle)

        #print("创建环境完成")

    def compute_reward(self):
        #print("开始计算reward")
        # retrieve environment observations from buffer
        dpx = self.obs_buf[:, 0]
        dpy = self.obs_buf[:, 1]
        dpz = self.obs_buf[:, 2]
        vx = self.obs_buf[:, 3]
        vy = self.obs_buf[:, 4]
        vz = self.obs_buf[:, 5]
        q0 = self.obs_buf[:, 6]
        q1 = self.obs_buf[:, 7]
        q2 = self.obs_buf[:, 8]
        q3 = self.obs_buf[:, 9]
        #phi = self.obs_buf[:, 6]
        #theta = self.obs_buf[:, 7]
        #psi = self.obs_buf[:, 8]

        #print("打印推力序列")
        #print(self.action_thrust)
        thrust = self.action_thrust.squeeze()
        #print("打印推力")
        #print(thrust)
        T1 = thrust[:, 0]
        T2 = thrust[:, 1]
        T3 = thrust[:, 2]
        T4 = thrust[:, 3]
        #print("打印T1")
        #print(T1)

        #print("打印dpx:")
        #print(dpx)
        #print("打印dpy:")
        #print(dpy)
        #print("打印dpz:")
        #print(dpz)

        # self.rew_buf[:], self.reset_buf[:] = compute_drone_reward(
        #     dpx, dpy, dpz, vx, vy, vz, phi, theta, psi,
        #     self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        # )
        self.rew_buf[:], self.reset_buf[:] = compute_drone_reward(
            dpx, dpy, dpz, vx, vy, vz, q0, q1, q2, q3, T1, T2, T3, T4,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )
        #print("打印self.rew_buf")
        #print(self.rew_buf)
        #print("打印self.reset_buf")
        #print(self.reset_buf)

        #print("计算reward完成")

    def compute_observations(self, env_ids=None):
        import copy
        #print("开始计算观测数据")
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        #self.root_angles = get_euler_xyz(self.root_quaternions)
        #self.root_angles_phi, self.root_angles_theta, self.root_angles_psi = get_euler_xyz(self.root_quaternions)
        # print("打印四元数")
        # print(self.root_quaternions)
        #self.root_angles = get_euler_xyz(self.root_quaternions)
        # self.root_angles = get_euler_xyz(torch.tensor([[0,0,0,1]]).repeat(8,1).to(self.device))
        #print("打印四元数")
        #print(self.root_quaternions)
        #print("打印欧拉角")
        #print(self.root_angles)
    
        targetx = 0.0
        targety = 0.0
        targetz = 2.0

        self.obs_buf[env_ids, 0] = targetx - self.root_positions[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = targety - self.root_positions[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 2] = targetz - self.root_positions[env_ids, 2].squeeze()

        self.obs_buf[env_ids, 3] = self.root_linvels[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 4] = self.root_linvels[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 5] = self.root_linvels[env_ids, 2].squeeze()

        self.obs_buf[env_ids, 6] = self.root_quaternions[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 7] = self.root_quaternions[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 8] = self.root_quaternions[env_ids, 2].squeeze()
        self.obs_buf[env_ids, 9] = self.root_quaternions[env_ids, 3].squeeze()

        '''
        self.obs_buf[env_ids, 6] = self.root_angles[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 7] = self.root_angles[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 8] = self.root_angles[env_ids, 2].squeeze()
        #self.obs_buf[env_ids, 6] = self.root_angles_ph
        #self.obs_buf[env_ids, 7] = self.root_angles_theta
        #self.obs_buf[env_ids, 8] = self.root_angles_psi
        #self.obs_buf[env_ids, 6] = self.root_angles[0]
        #self.obs_buf[env_ids, 7] = self.root_angles[1]
        #self.obs_buf[env_ids, 8] = self.root_angles[2]
        '''
        #print("打印观测序列")
        #print(self.obs_buf)
        
        #print("计算观测数据完成")
        
        return self.obs_buf

    def reset_idx(self, env_ids):
        #print("开始reset")
        '''
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        '''

        #######################
        num_resets = len(env_ids)
        #self.root_states[env_ids] = self.initial_root_states[env_ids]
        #pos_ini = [0, 0, 0.05]
        #q_ini = [1, 0, 0 , 0]

        self.root_tensor[env_ids, 0] = 0.0
        self.root_tensor[env_ids, 1] = 0.0
        self.root_tensor[env_ids, 2] = 2.0
        self.root_tensor[env_ids, 3] = 1.0
        self.root_tensor[env_ids, 4] = 0.0
        self.root_tensor[env_ids, 5] = 0.0
        self.root_tensor[env_ids, 6] = 0.0
        self.root_tensor[env_ids, 7] = 0.0
        self.root_tensor[env_ids, 8] = 0.0
        self.root_tensor[env_ids, 9] = 0.0
        self.root_tensor[env_ids, 10] = 0.0
        self.root_tensor[env_ids, 11] = 0.0
        self.root_tensor[env_ids, 12] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_tensor), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        #print("打印self.root_tensor")
        #print(self.root_tensor)


        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        #print("reset完成")

    def pre_physics_step(self, actions):
        #print("开始加载推力")
        #print("打印actions")
        #print(actions)


        actions_tensor = ((actions.to(self.device).unsqueeze(1) + 1) / 2) * self.max_push_effort * -1
        self.action_thrust = actions_tensor
        #print("打印actions_tensor")
        #print(actions_tensor)
        thrust = torch.zeros((self.num_envs, 5, 3), device=self.device, dtype=torch.float)
        thrust[:,1:,2] = actions_tensor[:,0,:]
        #print("打印thrust")
        #print(thrust)

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(thrust), None, gymapi.LOCAL_SPACE)
        
        '''
        forces = torch.zeros((self.num_envs, 5, 3), device=self.device, dtype=torch.float)
        forces[:,0,2] = 0
        forces[:,1,2] = -10
        forces[:,2,2] = -10
        forces[:,3,2] = -10
        forces[:,4,2] = -10
        #print(forces)
        
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE)
        '''

        #print("推力加载完成")


    def post_physics_step(self):
        #print("开始后处理")
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
        #print("后处理完成")

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_drone_reward(dpx, dpy, dpz, vx, vy, vz, q0, q1, q2, q3, T1, T2, T3, T4,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    target_dist = torch.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
    pos_reward = 1.0 / (1.0 + 10.0 * target_dist * target_dist)
    #pos_reward = -1 * (dpx * dpx + dpy * dpy + dpz * dpz) * 10
    #att_dist = torch.sqrt((1 - q0) * (1 - q0) + q1 * q1 + q2 * q2 + q3 * q3)
    #att_reward = 1.0 / (1.0 + 10.0 * att_dist * att_dist)
    att_reward = -1 * ((1 - q0) * (1 - q0) + q1 * q1 + q2 * q2 + q3 * q3)
    #att_reward = -1 * (phi * phi + theta * theta + psi * psi)
    thrust_average = (T1 + T2 + T3 + T4) / 4
    thrust_reward = -1 * ((T1-thrust_average) * (T1-thrust_average) + (T2-thrust_average) * (T2-thrust_average) + (T3-thrust_average) * (T3-thrust_average) + (T4-thrust_average) * (T4-thrust_average))
    reward = pos_reward + att_reward + 0.1 * thrust_reward
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 1, ones, die)
    #die = torch.where(phi > 0.785, ones, die)
    #die = torch.where(theta > 0.785, ones, die)
    #die = torch.where(root_positions[..., 2] < 0.3, ones, die)
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


# @torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    euler_angle = torch.zeros_like(q)
    roll = roll % (2*np.pi)
    pitch = pitch % (2*np.pi)
    yaw = yaw % (2*np.pi)
    euler_angle[:, 0] = roll[:]
    euler_angle[:, 1] = pitch[:]
    euler_angle[:, 2] = yaw[:]


    #return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)
    return euler_angle