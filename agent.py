from igibson.challenge.behavior_challenge import BehaviorChallenge

import json
import os

from igibson.examples.mp_replay.behavior_motion_primitive_env import MotionPrimitive
from igibson.utils.ig_logging import IGLogReader


def get_empty_hand(current_hands):
    if len(current_hands) == 0:
        return "right_hand"
    elif len(current_hands) == 1:
        return "left_hand" if list(current_hands.values())[0] == "right_hand" else "right_hand"

    raise ValueError("Both hands are full but you are trying to execute a grasp.")


def get_actions_from_segmentation(demo_data, only_first_from_multi_segment=True):
    print("Conversion of demo segmentation to motion primitives:")

    actions = []
    segmentation = demo_data["segmentations"]["flat"]["sub_segments"]

    # Convert the segmentation to a sequence of state changes.
    state_changes = []
    for segment in segmentation:
        state_records = segment["state_records"]
        if len(state_records) == 0:
            print("Found segment with no useful state changes: %r" % segment)
            continue
        elif len(state_records) > 1:
            if only_first_from_multi_segment:
                print("Found segment with multiple state changes, using the first: %r" % segment)
                state_records = [state_records[0]]
            else:
                print("Found segment with multiple state changes, using all: %r" % segment)

        state_changes.extend(state_records)

    # Now go through the state changes and convert them to actions
    for i, state_change in enumerate(state_changes):
        # Handle the combinations that we support.
        state_name = state_change["name"]
        state_value = state_change["value"]

        if state_name == "Open" and state_value is True:
            primitive = MotionPrimitive.OPEN
            target_object = state_change["objects"][0]
        elif state_name == "Open" and state_value is False:
            primitive = MotionPrimitive.CLOSE
            target_object = state_change["objects"][0]
        elif state_name == "InReachOfRobot" and state_value is True:
            # The primitives support automatic navigation to relevant objects.
            continue
        elif state_name == "InHandOfRobot" and state_value is True:
            # The primitives support automatic grasping of relevant objects.
            continue
        elif state_name == "Inside" and state_value is True:
            placed_object = state_change["objects"][0]
            target_object = state_change["objects"][1]
            primitive = MotionPrimitive.PLACE_INSIDE

            # Before the actual item is placed, insert a grasp request.
            actions.append((MotionPrimitive.GRASP, placed_object))
        elif state_name == "OnTop" and state_value is True:
            placed_object = state_change["objects"][0]
            target_object = state_change["objects"][1]
            primitive = MotionPrimitive.PLACE_ON_TOP

            # Before the actual item is placed, insert a grasp request.
            actions.append((MotionPrimitive.GRASP, placed_object))
        elif state_name == "OnFloor" and state_value is True:
            placed_object = state_change["objects"][0]
            target_object = state_change["objects"][1]
            primitive = MotionPrimitive.PLACE_ON_TOP

            # Before the actual item is placed, insert a grasp request.
            actions.append((MotionPrimitive.GRASP, placed_object))
        else:
            raise ValueError("Found a state change we can't process: %r" % state_change)

        # Append the action.
        action = (primitive, target_object)
        actions.append(action)

    for action in actions:
        print("%s(%s)" % action)

    print("Conversion complete.\n")
    return actions



class TaskPlanAgent:
    def __init__(self):
        self.actions = {}
        pass

    def generate_action_sequence(self, segmentation_path):
        # Load the segmentation of a demo for this task.
        with open(segmentation_path, "r") as f:
            selected_demo_data = json.load(f)

        # Get the actions from the segmentation
        actions = get_actions_from_segmentation(selected_demo_data)
        return actions

    def get_segmentation(self, task, task_id, scene_id):
        task_library = os.listdir('segmentations')
        for plan in task_library:
            if task in plan and task_id in plan and scene_id in plan:
                return plan
        return None

    def reset(self, env, env_config):
        self.env = env
        task = env_config["task"]
        task_id = env_config["task_id"]
        scene_id = env_config["scene_id"]
        segmentation_path = self.get_segmentation(task, task_id, scene_id)
        self.action_idx = 0
        if segmentation_path:
            self.actions = self.generate_action_sequence(segmentation_path)
        else:
            self.actions = None

    def act(self, _):
        if self.actions == None:
            return -1
        action_pair = self.actions[self.action_idx]

        print("Executing %s(%s)" % action_pair)
        primitive, obj_name = action_pair

        # Convert the action
        obj_id = next(i for i, obj in enumerate(self.env.addressable_objects) if obj.name == obj_name)
        action = int(primitive) * self.env.num_objects + obj_id

        self.action_idx += 1 

        return action


def main():

    challenge = BehaviorChallenge()
    challenge.submit(TaskPlanAgent())


if __name__ == "__main__":
    main()
