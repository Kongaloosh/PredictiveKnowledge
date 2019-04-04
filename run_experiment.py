#!/usr/bin/python
# coding: utf-8
from __future__ import print_function
import os
import peakAtState as peak
from display import *
from GVF import *
from GridWorld import *
from PIL import Image
from pysrc.prediction.network.td_network import *
from pysrc.prediction.network.off_policy_horde import HordeHolder, HordeLayer
from pysrc.control.control_agents import RandomAgent
from pysrc.function_approximation.StateRepresentation import Representation
from pysrc.prediction.cumulants.cumulant import Cumulant
import time

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)


class MinecraftCumulantTouch(Cumulant):

    def cumulant(self, obs):
        return obs['touchData']


class MinecraftCumulantPrediction(Cumulant):

    def __init__(self, i):
        self.prediction_index = i

    def cumulant(self, obs):
        return obs['predictions'][self.prediction_index]


class MinecraftHordeHolder(HordeHolder):
    """The minecraft experiment setup is slightly different from what we're used to doing, so I'm extending the state
    construction functionality such that it"""

    def __init__(self, layers, num_actions):
        """"Initializes a horde holder which is responsible for managing a hierarchical collection of predictions.
        Args:
            layers (list): a list which contains instances of HordeLayer.
            num_actions (int): the number of actions in the current domain.
        """
        super(MinecraftHordeHolder, self).__init__(layers=layers, num_actions=num_actions)

    def step(self, observations, policy=None, action=None, recurrance=False, skip=False, ude=False, remove_base=False, **vals):
        predictions = None
        for layer in self.layers:
            # Adds the most recent predictions to the observations; will be None if base layer.
            observations['predictions'] = predictions
            predictions =  layer.step(observations, policy, action)


class Foreground:

    def __init__(self, show_display=True, steps_before_updating_display=0, steps_before_prompting_for_action=0):
        """Initializes the experiment.
        Args:
            show_display (bool): a flag which determines whether the display is shown.
            steps_before_updating_display (int): the number of time-steps which are executed before presenting the display.
            steps_before_prompting_for_action (int): the number of time-steps which are executed before giving the user
            control.
        """
        self.show_display = show_display
        self.steps_before_prompting_for_action = steps_before_prompting_for_action
        self.steps_before_updating_display = steps_before_updating_display
        self.grid_world = GridWorld('model/grids', initial_x=1, initial_y=1)
        self.agent = RandomAgent(action_space=4, observation_space=0)
        if self.show_display:
            self.display = Display(self)
        self.gvfs = {}
        self.state = None
        self.old_state = None
        self.network = self.configure_gvfs_net()
        self.action_count = 0

    @staticmethod
    def configure_gvfs_net():
        """Follows the thought experiment from Ring (2016) to construct a multi-layer horde which gradually constructs
        predictions which are increasingly abstract. """

        layers = [] # where we store horde layers; creates a hierarchy.
        number_of_actions = 4
        # actions = "forward", "turn_left", "turn_right", "extend_hand"
        # =============================================================================================================
        # Layer 1 - Touch (T)
        # =============================================================================================================

        number_of_active_features = 400
        eligibility_decay = np.array([0.9])
        discounts = np.array([0])
        function_approximation = Representation()
        init_alpha = np.array([1./function_approximation.get_num_active()])
        policies = [[0, 0, 0, 1]]     # with probability 1, extend hand
        cumulant = [MinecraftCumulantTouch()]

        network = HordeLayer(
            function_approx=function_approximation,
            num_predictions=1,
            step_sizes=init_alpha,
            discounts=discounts,
            cumulants=cumulant,
            policies=policies,
            traces_lambda=eligibility_decay,
            protected_range=0,
        )
        layers.append(network)  # add the new layer to the collection

        # =============================================================================================================
        # Layer 2 - Touch Left (TL) and Touch Right (TR)
        # =============================================================================================================

        base_rep_dimension = PIXEL_FEATURE_LENGTH * NUMBER_OF_PIXEL_SAMPLES + DID_TOUCH_FEATURE_LENGTH
        policies = [[0, 1, 0, 0], [0, 0, 1, 0]]     # turn left and turn right
        cumulant = [MinecraftCumulantPrediction(0), MinecraftCumulantPrediction(0)]    # todo: what index?
        function_approximation = Representation(base_rep_dimension+1*PREDICTION_FEATURE_LENGTH)
        init_alpha = np.array(1. / function_approximation.get_num_active())
        network = HordeLayer(
            function_approx=function_approximation,
            num_predictions=2,
            step_sizes=init_alpha,
            discounts=discounts,
            cumulants=cumulant,
            policies=policies,
            traces_lambda=eligibility_decay,
            protected_range=0
        )
        layers.append(network)

        # =============================================================================================================
        # Layer 3 - Touch Behind
        # =============================================================================================================

        # Todo

        # =============================================================================================================
        # Layer 4 - Touch Adjacent (TA)
        # =============================================================================================================

        # Todo

        # =============================================================================================================
        # Layer 5 - Distance to touch adjacent (DTA)
        # Measures how many steps the agent is from being adjacent touch something.
        # * Note that because our agent only rotates 90 degrees at a time, this is basically the
        # number of steps to a wall. So the cumulant could be T. But we have the cumulant as TA instead
        # since this would allow for an agent whose rotations are not 90 degrees.
        # =============================================================================================================

        # Todo

        # =============================================================================================================
        # Layer 6 - Distance to Left (DTL), distance to right (DTR), distance back (DTB)
        # Measures how many steps to the left, or right, or behind,the agent is from a wall.
        # =============================================================================================================
        return MinecraftHordeHolder(layers, number_of_actions)

    def update_ui(self, action):
        """Re-draws the UI
        Args:
            action (int): the action taken this time-step.

        """
        # Create a voronoi image
        if self.state:
            frame = self.state['visionData']
            if self.show_display:
                voronoi = voronoi_from_pixels(pixels=frame, dimensions=(WIDTH, HEIGHT),
                                              pixelsOfInterest=self.state_representation.pointsOfInterest)
            # cv2.imshow('My Image', voronoi)
            # cv2.waitKey(0)

            if self.state is False:     # todo: should be None for first val, not bool.
                did_touch = False
            else:
                did_touch = self.state['touchData']

            # find the ground truth of the predictions.
            in_front = peak.isWallInFront(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            on_left = peak.isWallOnLeft(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            on_right = peak.isWallOnRight(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            is_behind = peak.isWallBehind(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            wall_adjacent = peak.isWallAdjacent(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            distance_to_adjacent = peak.distanceToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                           self.grid_world)
            distance_left = peak.distanceLeftToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                        self.grid_world)
            distance_right = peak.distanceRightToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                          self.grid_world)
            distance_back = peak.distanceBehindToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                          self.grid_world)
            wall_left_forward = peak.wallLeftForward(self.state['x'], self.state['y'], self.state['yaw'],
                                                     self.grid_world)

            # get the most recent predictions
            touch_prediction = self.network.layers[0].last_prediction
            # turn_left_and_touch_prediction = self.network.layers[1].last_prediction[0]
            turn_left_and_touch_prediction = 0
            # turn_right_and_touch_prediction = self.network.layers[1].last_prediction[1]
            turn_right_and_touch_prediction = 0
            # unimplemented, so zero...
            touch_behind_prediction = 0
            is_wall_adjacent_prediction = 0
            distance_to_adjacent_prediction = 0
            distance_left_prediction = 0
            distance_right_prediction = 0
            distance_back_prediction = 0
            wall_left_forward_prediction = 0

            game_image = Image.frombytes('RGB', (WIDTH, HEIGHT), bytes(frame))

            if self.show_display:
                if self.action_count > self.steps_before_updating_display:
                    self.display.update(voronoiImage=voronoi,
                                        gameImage=game_image,
                                        numberOfSteps=self.action_count,
                                        currentTouchPrediction=touch_prediction,
                                        wallInFront=in_front,
                                        didTouch=did_touch,
                                        turnLeftAndTouchPrediction=turn_left_and_touch_prediction,
                                        wallOnLeft=on_left,
                                        turnRightAndTouchPrediction=turn_right_and_touch_prediction,
                                        touchBehindPrediction=touch_behind_prediction,
                                        wallBehind=is_behind,
                                        touchAdjacentPrediction=is_wall_adjacent_prediction,
                                        wallAdjacent=wall_adjacent,
                                        wallOnRight=on_right,
                                        distanceToAdjacent=distance_to_adjacent,
                                        distanceToAdjacentPrediction=distance_to_adjacent_prediction,
                                        distanceToLeft=distance_left,
                                        distanceToLeftPrediction=distance_left_prediction,
                                        distanceToRight=distance_right,
                                        distanceToRightPrediction=distance_right_prediction,
                                        distanceBack=distance_back,
                                        distanceBackPrediction=distance_back_prediction,
                                        wallLeftForward=wall_left_forward,
                                        wallLeftForwardPrediction=wall_left_forward_prediction,
                                        action=action
                                        )

    def learn_from_behavior_policy_action(self):
        """Using the behaviour policy, selects an action. After selecting an action, updates the GVFs based on the
        action."""
        # todo: this is set as a variable in learn_from_action; we don't need to have two dependent calls...
        action = self.agent.get_action(state_prime=None)    # state doesn't matter; randint
        self.action_count += 1
        # If we've done 100 steps; pretty print the progress.
        if self.action_count % 100 == 0:
            print("Step " + str(self.action_count) + " ... ")
        observation = self.grid_world.take_action(action)
        # Do the learning
        self.network.step(observation, self.agent.get_policy(observation=None), action)
        # Update our display (for debugging and progress reporting)
        self.update_ui(action)

    def start(self):
        """Initializes the plotter and runs the experiment."""
        # Loop until mission ends:
        while self.action_count < self.steps_before_prompting_for_action:
            # Select and send action. Need to sleep to give time for simulator to respond
            self.learn_from_behavior_policy_action()
            time.sleep(1)
        self.display.root.mainloop()
        print("Mission ended")
        # Mission has ended.


if __name__ == "__main__":
    # fg.read_gvf_weights()
    fg = Foreground(show_display=True, steps_before_updating_display=0, steps_before_prompting_for_action=12000)
    fg.start()
