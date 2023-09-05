from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend._sim_cffi import lib
from pyrep import PyRep

from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.robot import Robot
from rlbench.const import colors


class PickAndLiftSimple(Task):

    def __init__(self, pyrep: PyRep, robot: Robot, name: str = None):
        super().__init__(pyrep,robot,name)
        self.colors = colors[:5]
    def init_task(self) -> None:

        self.target_pos = ["left","right"] 

		# objects
        self.target_block = Shape('pick_and_lift_target') 
        self.distractor = Shape('stack_blocks_distractor')
        self.register_graspable_objects([self.target_block])

		# spawn boundaries
        self.boundary_left = SpawnBoundary([Shape('pick_and_lift_boundary_left')])
        self.boundary_right = SpawnBoundary([Shape('pick_and_lift_boundary_right')])
        
		# success detector
        self.success_detector = ProximitySensor('pick_and_lift_success')
		
		# success conditions
        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.target_block, self.success_detector)
        ])
        self.register_success_conditions([cond_set])


    def init_episode(self, index: int) -> List[str]:
        
        # change size
        # lib.simScaleObject(self.target_block._handle,0.5,0.5,0.5,0)
        block_color_name, block_rgb = self.colors[index]

        self.target_block.set_color(block_rgb)
        color_choice = np.random.choice( 
            list(range(index)) + list(range(index + 1, len(self.colors))),
            size=1, replace=False)[0]
        name, rgb = self.colors[color_choice]

        self.distractor.set_color(rgb)

        self.boundary_left.clear()
        self.boundary_right.clear()
        target_pos = np.random.choice(self.target_pos)
        if target_pos == "left":
            self.boundary_left.sample(
                self.target_block, min_rotation=(0.0, 0.0, 0.0),
                max_rotation=(0.0, 0.0, 0.0))           
            self.boundary_right.sample(
                self.distractor, min_rotation=(0.0, 0.0, 0.0),
                max_rotation=(0.0, 0.0, 0.0))
        elif target_pos == "right":
            self.boundary_left.sample(
                self.distractor, min_rotation=(0.0, 0.0, 0.0),
                max_rotation=(0.0, 0.0, 0.0))           
            self.boundary_right.sample(
                self.target_block, min_rotation=(0.0, 0.0, 0.0),
                max_rotation=(0.0, 0.0, 0.0))
            
        # fixed success target            
        # self.boundary_left.sample(
        #     self.success_detector, min_rotation=(0.0, 0.0, 0.0),
        #     max_rotation=(0.0, 0.0, 0.0))

        
        print("return")
        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def variation_count(self) -> int:
        return len(self.colors)

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.concatenate([self.target_block.get_position(), self.success_detector.get_position()], 0)

    def is_static_workspace(self) -> bool:
            return True
        