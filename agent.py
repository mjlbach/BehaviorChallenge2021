import argparse

from igibson.challenge.behavior_challenge import BehaviorChallenge

import numpy as np

import json
import os
import time

import yaml

import igibson
from igibson.envs.behavior_mp_env import ActionPrimitives, BehaviorMPEnv
from igibson.utils.ig_logging import IGLogReader


def get_empty_hand(current_hands):
    if len(current_hands) == 0:
        return "right_hand"
    elif len(current_hands) == 1:
        return "left_hand" if list(current_hands.values())[0] == "right_hand" else "right_hand"

    raise ValueError("Both hands are full but you are trying to execute a grasp.")


def get_actions_from_segmentation(demo_data):
    print("Conversion of demo segmentation to motion primitives:")

    hand_by_object = {}
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
            print("Found segment with multiple state changes, using the first: %r" % segment)

        state_change = state_records[0]
        state_changes.append(state_change)

    # Now go through the state changes and convert them to actions
    for i, state_change in enumerate(state_changes):
        # Handle the combinations that we support.
        state_name = state_change["name"]
        state_value = state_change["value"]

        # TODO(replayMP): Here we compute grasps based on the InHand state. Ditch this and simply do a single-hand
        # grasp on the object we will manipulate next. That way it will be fetch-compatible.
        if state_name == "Open" and state_value is True:
            primitive = ActionPrimitives.OPEN
            target_object = state_change["objects"][0]
        elif state_name == "Open" and state_value is False:
            primitive = ActionPrimitives.CLOSE
            target_object = state_change["objects"][0]
        elif state_name == "InReachOfRobot" and state_value is True:
            # The primitives support automatic navigation to relevant objects.
            continue
        elif state_name == "InHandOfRobot" and state_value is True:
            target_object = state_change["objects"][0]

            # Check that we do something with this object later on, otherwise don't grasp it.
            is_used = False
            for future_state_change in state_changes[i + 1 :]:
                # We should only grasp the moved object in these cases.
                if future_state_change["objects"][0] != target_object:
                    continue

                if future_state_change["name"] == "InHandOfRobot" and future_state_change["value"] is True:
                    # This object is re-grasped later. No need to look any further than that.
                    break

                # We only care about Inside and OnTop use cases later.
                if future_state_change["name"] not in ("Inside", "OnTop") or future_state_change["value"] is False:
                    continue

                # This is a supported use case so we approve the grasp.
                is_used = True
                break

            # If the object is not used in the future, don't grasp it.
            if not is_used:
                continue

            hand = get_empty_hand(hand_by_object)
            hand_by_object[target_object] = hand
            primitive = ActionPrimitives.LEFT_GRASP if hand == "left_hand" else ActionPrimitives.RIGHT_GRASP
        elif state_name == "Inside" and state_value is True:
            placed_object = state_change["objects"][0]
            target_object = state_change["objects"][1]
            if placed_object not in hand_by_object:
                print(
                    "Placed object %s in segment %d not currently grasped. Maybe some sort of segmentation error?"
                    % (placed_object, i)
                )
                continue
            hand = hand_by_object[placed_object]
            del hand_by_object[placed_object]
            primitive = (
                ActionPrimitives.LEFT_PLACE_INSIDE if hand == "left_hand" else ActionPrimitives.RIGHT_PLACE_INSIDE
            )
        elif state_name == "OnTop" and state_value is True:
            placed_object = state_change["objects"][0]
            target_object = state_change["objects"][1]
            if placed_object not in hand_by_object:
                print(
                    "Placed object %s in segment %d not currently grasped. Maybe some sort of segmentation error?"
                    % (placed_object, i)
                )
                continue
            hand = hand_by_object[placed_object]
            del hand_by_object[placed_object]
            primitive = ActionPrimitives.LEFT_PLACE_ONTOP if hand == "left_hand" else ActionPrimitives.RIGHT_PLACE_ONTOP
        else:
            raise ValueError("Found a state change we can't process: %r" % state_change)

        # Append the action.
        action = (primitive, target_object)
        actions.append(action)
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
        self.actions = get_actions_from_segmentation(selected_demo_data)


    def reset(self, env, env_config):
        task = env_config["task"]
        task_id = env_config["task_id"]
        scene_id = env_config["scene_id"]
        import pdb; pdb.set_trace()
        self.action_idx = 0
        self.actions = self.generate_action_sequence(segmentation_path)
        self.env = env
        pass

    def act(self, _):
        # env.robots[0].set_position_orientation([0, 0, 0.7], [0, 0, 0, 1])
        action_pair = self.actions[self.action_idx]
        # try:
        print("Executing %s(%s)" % action_pair)
        primitive, obj_name = action_pair

        # Convert the action
        obj_id = next(i for i, obj in enumerate(self.env.addressable_objects) if obj.name == obj_name)
        action = int(primitive) * self.env.num_objects + obj_id

        self.action_idx += 1 

        return action


def main():

    challenge = BehaviorChallenge()
    challenge.submit(TaskPlanAgent)


if __name__ == "__main__":
    main()
