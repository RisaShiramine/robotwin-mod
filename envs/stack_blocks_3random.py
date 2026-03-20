from ._base_task import Base_Task
from .utils import *
import sapien
import math


class stack_blocks_3random(Base_Task):

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

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.block3 = create_block(block_pose_lst[2], (0, 0, 1))

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)

        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

        # will be filled in play_once (episode-specific order)
        self.stack_order = [self.block1, self.block2, self.block3]

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        blocks = [
            (self.block1, "red block"),
            (self.block2, "green block"),
            (self.block3, "blue block"),
        ]

        perm = np.random.permutation(len(blocks))
        ordered = [blocks[i] for i in perm]

        self.stack_order = [ordered[0][0], ordered[1][0], ordered[2][0]]

        arm_tag1 = self.pick_and_place_block(ordered[0][0])
        arm_tag2 = self.pick_and_place_block(ordered[1][0])
        arm_tag3 = self.pick_and_place_block(ordered[2][0])

        self.info["info"] = {
            "{A}": ordered[0][1],
            "{B}": ordered[1][1],
            "{C}": ordered[2][1],
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag3),
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
        bottom, middle, top = self.stack_order
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
