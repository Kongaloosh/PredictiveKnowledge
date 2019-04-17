#!/usr/bin/python
# coding: utf-8
from __future__ import print_function
import os
import peakAtState as peak
from display import *
from GridWorld import *
from PIL import Image
from pysrc.prediction.network.td_network import *
from pysrc.prediction.network.off_policy_horde import HordeHolder, HordeLayer
from pysrc.control.control_agents import RandomAgent
from pysrc.function_approximation.StateRepresentation import Representation, TrackingRepresentation
from pysrc.prediction.cumulants.cumulant import Cumulant
from matplotlib import pyplot
from pysrc.prediction.off_policy.gtd import *
from os import system

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)


class MinecraftAgent(RandomAgent):
    """
        A random agent which chooses actions equiprobably regardless of observations.
    """

    def __init__(self, observation_space, action_space, **vals):
        super(RandomAgent, self).__init__(observation_space, action_space)
        self.state_prime = None
        self.i = 0

    @staticmethod
    def __str__():
        return "RandomAgent"

    def terminal_step(self, reward):
        pass

    def initialize_episode(self):
        pass

    def get_action(self, state_prime):
        action = np.random.randint(self.action_space)
        while action == 1:
            action = np.random.randint(self.action_space)
        return action

    def get_policy(self, observation):
        return np.ones(self.action_space)/self.action_space

    def step(self, observation, reward):
        pass


class MinecraftCumulantTouch(Cumulant):

    def cumulant(self, obs):
        return obs['touchData']


class MinecraftCumulantPrediction(Cumulant):

    def __init__(self, i):
        self.prediction_index = i

    def cumulant(self, obs):
        return obs['predictions'][self.prediction_index]


class MinecraftHordeLayer(HordeLayer):

    def step(self, observations, policy, action, remove_base=False, terminal_step=False, **vals):
        """Update the Network
        Args:
            observations (list): real-valued list of observations from the environment.
            policy (list): list of length num_actions; the policy of the control policy for the given state.
        Returns:
            predictions (list): the predictions for each GVF given the observations and policy.
        """
        # get the next feature vector
        phi_next = self.function_approximation.get_features(observations)
        if type(self.last_phi) is np.ndarray:
            discounts = np.array([discount.gamma(observations) for discount in self.discounts])

            if terminal_step:
                discounts = np.zeros(self.discounts.shape)
            # calculate importance sampling
            rho = (self.policies/policy)[:, action]
            # update the traces based on the new visitation
            self.eligibility_traces = accumulate(self.eligibility_traces, discounts, self.traces_lambda, self.last_phi, rho)
            # calculate the new cumulants
            current_cumulants = np.array([cumulant.cumulant(observations) for cumulant in self.cumulants])
            # get a vector of TD errors corresponding to the performance.
            td_error = calculate_temporal_difference_error(self.weights, current_cumulants, discounts, phi_next,
                                                           self.last_phi)
            # update the weights based on the caluculated TD error
            self.weights = update_weights(td_error, self.eligibility_traces, self.weights, discounts, self.traces_lambda, self.step_sizes, self.last_phi, self.bias_correction)
            # update bias correction term
            self.bias_correction = update_h_trace(self.bias_correction, td_error, self.step_size_bias_correction
                                                  , self.eligibility_traces, self.last_phi)


            # maintain verifiers
            self.rupee, self.tau, self.eligibility_avg = \
                update_rupee(
                    beta_naught=self.rupee_beta,
                    tau=self.tau,
                    delta_e=self.eligibility_avg,
                    h=self.bias_correction,
                    e=self.eligibility_traces,
                    delta=td_error,
                    alpha=self.step_sizes,
                    phi=self.last_phi
                )
            self.ude, self.delta_avg, self.delta_var = update_ude(
                self.ude_beta,
                self.delta_avg,
                self.delta_var,
                td_error
            )
            self.avg_error = self.avg_error * 0.9 + 0.1 * np.abs(td_error)

        self.last_phi = phi_next
        self.last_prediction = np.inner(self.weights, phi_next)
        return self.last_prediction


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
        self.state = None
        self.rupee = None
        self.ude = None
        self.prediction = None

    def step(self, observations, policy=None, action=None, recurrance=False, skip=False, ude=False, remove_base=False, **vals):
        predictions = None
        all_predictions = []
        for layer in self.layers:
            # Adds the most recent predictions to the observations; will be None if base layer.
            observations['predictions'] = predictions
            predictions = layer.step(observations, policy, action)
            all_predictions.append(predictions.tolist())

        rupee = self.get_rupee()
        ude = self.get_ude()
        try:
            for i in range(len(rupee)):
                self.rupee[i].append(rupee[i])
                self.ude[i].append(ude[i])
                self.prediction[i].append(all_predictions[i])
        except TypeError:
            self.rupee = []
            self.ude = []
            self.prediction = []
            for i in range(len(rupee)):
                self.rupee.append([])
                self.ude.append([])
                self.prediction.append([])
                self.rupee[i].append(rupee[i])
                self.ude[i].append(ude[i])
                self.prediction[i].append(all_predictions[i])
        self.state = observations


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
        self.tracking_network = self.configure_tracking_gvfs()
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
        init_alpha = np.array([0.3 /function_approximation.get_num_active()])
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
        discounts = np.array([0, 0])
        cumulant = [MinecraftCumulantPrediction(0), MinecraftCumulantPrediction(0)]    # todo: what index?
        function_approximation = Representation(base_rep_dimension+1*PREDICTION_FEATURE_LENGTH)
        init_alpha = np.array(0.3 / function_approximation.get_num_active())
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

    @staticmethod
    def configure_tracking_gvfs():
        """Follows the thought experiment from Ring (2016) to construct a multi-layer horde which gradually constructs
        predictions which are increasingly abstract. """

        layers = []  # where we store horde layers; creates a hierarchy.
        number_of_actions = 4
        # actions = "forward", "turn_left", "turn_right", "extend_hand"
        # =============================================================================================================
        # Layer 1 - Touch (T)
        # =============================================================================================================

        number_of_active_features = 400
        eligibility_decay = np.array([0.9])
        discounts = np.array([0])
        function_approximation = Bias()
        init_alpha = np.array([0.3 / function_approximation.get_num_active()])
        policies = [[0, 0, 0, 1]]  # with probability 1, extend hand
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
        policies = [[0, 1, 0, 0], [0, 0, 1, 0]]  # turn left and turn right
        discounts = np.array([0, 0])
        cumulant = [MinecraftCumulantPrediction(0), MinecraftCumulantPrediction(0)]
        function_approximation = Representation(base_rep_dimension + 1 * PREDICTION_FEATURE_LENGTH)
        init_alpha = np.array(0.1 / function_approximation.get_num_active())
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
                                              pixelsOfInterest=self.network.layers[0].function_approximation.pointsOfInterest)
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
            touch_prediction = self.network.layers[0].last_prediction[0]
            track_touch_prediction = self.tracking_network.layers[0].last_prediction[0]

            turn_left_and_touch_prediction = self.network.layers[1].last_prediction[0]
            track_turn_left_and_touch_prediction = self.tracking_network.layers[1].last_prediction[0]

            # turn_left_and_touch_prediction = 0

            turn_right_and_touch_prediction = self.network.layers[1].last_prediction[1]
            track_turn_right_and_touch_prediction = self.tracking_network.layers[1].last_prediction[1]

            # turn_right_and_touch_prediction = 0
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
                                        action=action,
                                        track_touch_prediction=track_touch_prediction,
                                        track_turn_left_and_touch_prediction=track_turn_left_and_touch_prediction,
                                        track_turn_right_and_touch_prediction=track_turn_right_and_touch_prediction
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
        self.tracking_network.step(observation, self.agent.get_policy(observation=None), action)
        self.state = self.network.state
        # Update our display (for debugging and progress reporting)
        self.update_ui(action)

    def get_true_values(self):
        true_values = []
        in_front = peak.isWallInFront(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)

        true_values.append([in_front])

        on_left = peak.isWallOnLeft(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
        on_right = peak.isWallOnRight(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)

        true_values.append([on_left, on_right])

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
        return true_values

    def start(self):
        """Initializes the plotter and runs the experiment."""
        print("Mission ended")

        track_rupee = []
        track_ude = []
        track_prediction = []

        exp_rupee = []
        exp_ude = []
        exp_prediction = []

        true_values = []

        for seed in [
            8995, 6553, 2514, 6629, 7381, 1590, 1588, 2585, 1083,  822,
            438, 3674, 8768, 8891, 6448, 5719, 5134, 8341, 5981,
            3623, 6994, 1653, 5417, 6542,
            4868, 9414, 6632, 1852, 1788, 3348
        ]:
            random.seed(seed)
            np.random.seed(seed)

            while self.action_count < self.steps_before_prompting_for_action:
                self.learn_from_behavior_policy_action()
                true_values_now = self.get_true_values()
                try:
                    for i in range(len(true_values_now)):
                        true_values[i].append(true_values_now[i])
                except IndexError:
                    for i in range(len(true_values_now)):
                        true_values.append([])
                        true_values[i].append(true_values_now[i])

            # pyplot.plot(np.array(self.network.rupee[0]))
            self.action_count = 0
            # dump network rupee and error
            # dump the weights of the predictions & serialize function approximators
            try:
                for i in range(len(self.network.rupee)):
                    exp_rupee[i].append(self.network.rupee[i])
                    exp_ude[i].append(self.network.ude[i])
                    exp_prediction[i].append(self.network.prediction[i])

            except IndexError:
                for i in range(len(self.network.rupee)):
                    exp_rupee.append([])
                    exp_ude.append([])
                    exp_prediction.append([])

                    track_rupee.append([])
                    track_ude.append([])
                    track_prediction.append([])

                    exp_rupee[i].append(self.network.rupee[i])
                    exp_ude[i].append(self.network.ude[i])
                    exp_prediction[i].append(self.network.prediction[i])

                    track_rupee[i].append(self.tracking_network.rupee[i])
                    track_ude[i].append(self.tracking_network.ude[i])
                    track_prediction[i].append(self.tracking_network.prediction[i])

            self.network = self.configure_gvfs_net()  # reset network
            self.tracking_network = self.configure_tracking_gvfs()
        with open('results/experiment_results.pkl', "wb") as f:
            pickle.dump(
                {
                    'predictive_rupee': exp_rupee,
                    'predictive_ude': exp_ude,
                    'predictive_predictions' : exp_prediction,
                    'tracking_rupee': track_rupee,
                    'tracking_ude' : track_ude,
                    'tracking_predictions' : track_prediction,
                    'environment_values' : true_values,
                },
                f
            )

        for i in range(len(exp_rupee)):
            exp_prediction[i] = np.average(exp_prediction[i], axis=0)
            exp_ude[i] = np.average(exp_ude[i], axis=0)
            exp_rupee[i] = np.average(exp_rupee[i], axis=0)
            track_prediction[i] = np.average(track_prediction[i], axis=0)
            track_ude[i] = np.average(track_ude[i], axis=0)
            track_rupee[i] = np.average(track_rupee[i], axis=0)

        system('say your experiment is finished')
        print("plotting")
        for i in [0,1]:
            pyplot.plot(exp_rupee[i], label="predictor")
            pyplot.plot(track_rupee[i], alpha=0.2, label="tracker")
            pyplot.legend()
            pyplot.show()



if __name__ == "__main__":
    # fg.read_gvf_weights()
    fg = Foreground(show_display=False, steps_before_updating_display=1000, steps_before_prompting_for_action=25000)
    fg.start()
