from ._base_task import Base_Task
from .utils import *
import sapien
import math


class stack_3blocks_random(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        block_pose_lst = []

        for _ in range(3):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(pose):
                for exist_pose in block_pose_lst:
                    if np.sum(pow(pose.p[:2] - exist_pose.p[:2], 2)) < 0.01:
                        return False
                return True

            while (abs(block_pose.p[0]) < 0.05
                   or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
                   or not check_block_pose(block_pose)):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            block_pose_lst.append(deepcopy(block_pose))

        color_options = [
            ((1, 0, 0), "red"),
            ((0, 1, 0), "green"),
            ((0, 0, 1), "blue"),
        ]

        # Randomize RGB distribution across the 3 spawn poses
        color_ids = np.random.permutation(len(color_options))
        pose_color_pairs = [(block_pose_lst[i], color_options[color_ids[i]]) for i in range(3)]

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.blocks = []
        self.block_color_names = []
        for pose, (rgb, name) in pose_color_pairs:
            self.blocks.append(create_block(pose, rgb))
            self.block_color_names.append(name)

        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.05)

        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

        # Episode-specific stacking order will be set in play_once
        self.stack_order_indices = [0, 1, 2]

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        # Ensure 6 stacking orders are covered by cycling with ep_num
        perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        perm = perms[int(self.ep_num) % 6]
        self.stack_order_indices = list(perm)

        idx_bottom, idx_middle, idx_top = perm
        bottom = self.blocks[idx_bottom]
        middle = self.blocks[idx_middle]
        top = self.blocks[idx_top]

        arm_tag_a = self.pick_and_place_block(bottom)
        arm_tag_b = self.pick_and_place_block(middle)
        arm_tag_c = self.pick_and_place_block(top)

        self.info["info"] = {
            "{A}": f"{self.block_color_names[idx_bottom]} block",
            "{B}": f"{self.block_color_names[idx_middle]} block",
            "{C}": f"{self.block_color_names[idx_top]} block",
            "{a}": str(arm_tag_a),
            "{b}": str(arm_tag_b),
            "{c}": str(arm_tag_c),
        }
        return self.info

    def pick_and_place_block(self, block: Actor):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        if self.last_actor is None:
            target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        else:
            target_pose = self.last_actor.get_functional_point(1)

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def check_success(self):
        idx_bottom, idx_middle, idx_top = self.stack_order_indices
        bottom = self.blocks[idx_bottom]
        middle = self.blocks[idx_middle]
        top = self.blocks[idx_top]

        bottom_pose = bottom.get_pose().p
        middle_pose = middle.get_pose().p
        top_pose = top.get_pose().p

        eps = [0.025, 0.025, 0.012]

        return (
            np.all(abs(middle_pose - np.array(bottom_pose[:2].tolist() + [bottom_pose[2] + 0.05])) < eps)
            and np.all(abs(top_pose - np.array(middle_pose[:2].tolist() + [middle_pose[2] + 0.05])) < eps)
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
