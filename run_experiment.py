#!/usr/bin/python
# coding: utf-8
from __future__ import print_function
import os
import peakAtState as peak
from display import *
from GVF import *
from GridWorld import *
from PIL import Image

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools

    print = functools.partial(print, flush=True)


def did_touch_cumulant(phi):
    """Checks to see if the the touch.
    Args:
        phi (ndarray): a feature-vector which describes the current state.
    Returns:
        did_touch (int): whether the agent touched a wall.
    """
    return phi[-1:]


def did_touch_gamma(phi):
    return 0


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
        self.grid_world = GridWorld('model/grids', initialX=1, initialY=1)
        self.behavior_policy = BehaviorPolicy()

        if self.show_display:
            self.display = Display(self)
        self.gvfs = {}
        self.configure_gvfs()
        self.state_representation = StateRepresentation(self.gvfs)
        self.state = False
        self.old_state = False
        self.phi = self.state_representation.get_empty_phi()
        self.old_phi = self.state_representation.get_empty_phi()

        self.action_count = 0
        self.action = None

    def save_gvf_weights(self):
        """Saves the weights of the GVFs to a """
        for name, gvf in self.gvfs.items():
            gvf.saveWeightsToPickle('weights/' + str(gvf.name))
        self.state_representation.save_points_of_interest('weights/pointsofinterest')

    def read_gvf_weights(self):
        """For a collection of GVFs, checks to see if there's a file which matches the """
        for name, gvf in self.gvfs.items():
            print("Reading weights for " + str(name))
            gvf.readWeightsFromPickle('weights/' + str(gvf.name))
        self.state_representation.read_points_of_interest('weights/pointsofinterest')

    @staticmethod
    def did_touch_gamma(phi):
        """"""
        return 0

    def configure_gvfs(self, simple_phi=False):
        """Configures the GVFs horde based on Mark Ring's thought experiment."""
        touch_threshold = 0.8  # The prediction value before it is considered to be true.

        # =============================================================================================================
        # Layer 1 - Touch (T)
        # =============================================================================================================

        touch_gvf = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                        alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True, name="T")

        touch_gvf.cumulant = did_touch_cumulant
        touch_gvf.policy = self.behavior_policy.extendHandPolicy

        touch_gvf.gamma = did_touch_gamma
        self.gvfs[touch_gvf.name] = touch_gvf

        # =============================================================================================================
        # Layer 2 - Touch Left (TL) and Touch Right (TR)
        # =============================================================================================================

        turn_left_gvf = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                            alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True, name="TL")

        turn_left_gvf.cumulant = self.gvfs['T'].prediction
        turn_left_gvf.policy = self.behavior_policy.turnLeftPolicy
        turn_left_gvf.gamma = did_touch_gamma
        self.gvfs[turn_left_gvf.name] = turn_left_gvf

        turn_right_gvf = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                             alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True, name="TR")

        turn_right_gvf.cumulant = self.gvfs['T'].prediction
        turn_right_gvf.policy = self.behavior_policy.turnRightPolicy
        turn_right_gvf.gamma = did_touch_gamma
        self.gvfs[turn_right_gvf.name] = turn_right_gvf

        # =============================================================================================================
        # Layer 3 - Touch Behind
        # =============================================================================================================

        touch_behind_gvf = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                                   alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True,
                                   name="TB")

        touch_behind_gvf.cumulant = self.gvfs['TR'].prediction
        touch_behind_gvf.policy = self.behavior_policy.turnRightPolicy
        touch_behind_gvf.gamma = did_touch_gamma
        self.gvfs[touch_behind_gvf.name] = touch_behind_gvf

        # =============================================================================================================
        # Layer 4 - Touch Adjacent (TA)
        # =============================================================================================================

        touch_adjacent_gvf = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                                 alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True,
                                 name="TA")

        def ta_learn(lastState, action, newState):
            return

        def ta_prediction(phi):
            predict = max(
                [self.gvfs['T'].prediction(phi), self.gvfs['TL'].prediction(phi), self.gvfs['TR'].prediction(phi),
                 self.gvfs['TB'].prediction(phi)])
            return predict

        touch_adjacent_gvf.prediction = ta_prediction
        touch_adjacent_gvf.learn = ta_learn

        self.gvfs[touch_adjacent_gvf.name] = touch_adjacent_gvf

        # =============================================================================================================
        # Layer 5 - Distance to touch adjacent (DTA)
        # Measures how many steps the agent is from being adjacent touch something.
        # * Note that because our agent only rotates 90 degrees at a time, this is basically the
        # number of steps to a wall. So the cumulant could be T. But we have the cumulant as TA instead
        # since this would allow for an agent whose rotations are not 90 degrees.
        # =============================================================================================================

        distanceToTouchAdjacentGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                                         alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES),
                                         is_off_policy=True, name="DTA")

        def distanceToTouchAdjacentCumulant(phi):
            return 1.0

        distanceToTouchAdjacentGVF.cumulant = distanceToTouchAdjacentCumulant
        distanceToTouchAdjacentGVF.policy = self.behavior_policy.moveForwardPolicy

        def distanceToTouchAdjacentGamma(phi):
            prediction = self.gvfs['T'].prediction(phi)  # TODO - change to self.gvfs['TA'].prediction() after testing
            if prediction > touch_threshold:
                return 0
            else:
                return 1

        distanceToTouchAdjacentGVF.gamma = distanceToTouchAdjacentGamma

        self.gvfs[distanceToTouchAdjacentGVF.name] = distanceToTouchAdjacentGVF

        # =============================================================================================================
        # Layer 6 - Distance to Left (DTL), distance to right (DTR), distance back (DTB)
        # Measures how many steps to the left, or right, or behind,the agent is from a wall.
        # =============================================================================================================

        distanceToLeftGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                                alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True,
                                name="DTL")

        def distanceToLeftCumulant(phi):
            return self.gvfs['DTA'].prediction(phi)

        distanceToLeftGVF.cumulant = distanceToLeftCumulant

        def distanceToLeftGamma(phi):
            return 0

        distanceToLeftGVF.gamma = distanceToLeftGamma
        distanceToLeftGVF.policy = self.behavior_policy.turnLeftPolicy
        self.gvfs[distanceToLeftGVF.name] = distanceToLeftGVF

        # Distance to Right GVF
        distanceToRightGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                                 alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True,
                                 name="DTR")

        def distanceToRightCumulant(phi):
            return self.gvfs['DTA'].prediction(phi)

        distanceToRightGVF.cumulant = distanceToRightCumulant

        def distanceToRightGamma(phi):
            return 0

        distanceToRightGVF.gamma = distanceToRightGamma
        distanceToRightGVF.policy = self.behavior_policy.turnRightPolicy
        self.gvfs[distanceToRightGVF.name] = distanceToRightGVF

        # Distance behind GVF
        distanceToBackGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                                alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True,
                                name="DTB")

        def distanceToBackCumulant(phi):
            return self.gvfs['DTR'].prediction(phi)

        distanceToBackGVF.cumulant = distanceToBackCumulant

        def distanceToBackGamma(phi):
            return 0

        distanceToBackGVF.gamma = distanceToBackGamma
        distanceToBackGVF.policy = self.behavior_policy.turnRightPolicy
        self.gvfs[distanceToBackGVF.name] = distanceToBackGVF

        # Wall left forward GVF (ie. how many steps the agent can take forward while keeping a wall on the left
        wallLeftForwardGVF = GVF(featureVectorLength=TOTAL_FEATURE_LENGTH,
                                 alpha=0.10 / (NUM_IMAGE_TILINGS * NUMBER_OF_PIXEL_SAMPLES), is_off_policy=True,
                                 name="WLF")

        def wallLeftForwardCumulant(phi):
            wallLeftPredict = self.gvfs['TL'].prediction(phi)
            if wallLeftPredict > touch_threshold:
                return 1.0
            else:
                return 0.0

        wallLeftForwardGVF.cumulant = wallLeftForwardCumulant

        def wallLeftGamma(phi):
            wallLeftPredict = self.gvfs['TL'].prediction(phi)
            if wallLeftPredict > touch_threshold:
                return 0.9
            else:
                return 0.0

        wallLeftForwardGVF.gamma = wallLeftGamma
        wallLeftForwardGVF.policy = self.behavior_policy.moveForwardPolicy
        self.gvfs[wallLeftForwardGVF.name] = wallLeftForwardGVF


    def learn(self):
        """Updates all the GVFs in the horde."""
        for name, gvf in self.gvfs.items():
            gvf.learn(lastState=self.old_phi, action=self.action, newState=self.phi)


    def update_ui(self):
        # Create a voronoi image
        frameError = False
        try:
            frame = self.state['visionData']
        except:
            frameError = True
            print("Error gettnig frame")

        if not frameError:
            # rgb = self.s
            if self.show_display:
                voronoi = voronoi_from_pixels(pixels=frame, dimensions=(WIDTH, HEIGHT),
                                              pixelsOfInterest=self.state_representation.pointsOfInterest)
            # cv2.imshow('My Image', voronoi)
            # cv2.waitKey(0)

            if self.state == False:
                didTouch = False
            else:
                didTouch = self.state['touchData']

            inFront = peak.isWallInFront(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            touchPrediction = self.gvfs['T'].prediction(self.phi)

            gameImage = Image.frombytes('RGB', (WIDTH, HEIGHT), bytes(frame))

            onLeft = peak.isWallOnLeft(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            turnLeftAndTouchPrediction = self.gvfs['TL'].prediction(self.phi)

            onRight = peak.isWallOnRight(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            turnRightAndtouchPrediction = self.gvfs['TR'].prediction(self.phi)

            isBehind = peak.isWallBehind(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            touchBehindPrediction = self.gvfs['TB'].prediction(self.phi)

            wallAdjacent = peak.isWallAdjacent(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            isWallAdjacentPrediction = self.gvfs['TA'].prediction(self.phi)

            distanceToAdjacent = peak.distanceToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                         self.grid_world)
            distanceToAdjacentPrediction = self.gvfs['DTA'].prediction(self.phi)

            distanceLeft = peak.distanceLeftToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                       self.grid_world)
            distanceLeftPrediction = self.gvfs['DTL'].prediction(self.phi)

            distanceRight = peak.distanceRightToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                         self.grid_world)
            distanceRightPrediction = self.gvfs['DTR'].prediction(self.phi)

            distanceBack = peak.distanceBehindToAdjacent(self.state['x'], self.state['y'], self.state['yaw'],
                                                         self.grid_world)
            distanceBackPrediction = self.gvfs['DTB'].prediction(self.phi)

            wallLeftForward = peak.wallLeftForward(self.state['x'], self.state['y'], self.state['yaw'], self.grid_world)
            wallLeftForwardPrediction = self.gvfs['WLF'].prediction(self.phi)

            if self.show_display:
                if self.action_count > self.steps_before_updating_display:
                    self.display.update(voronoiImage=voronoi,
                                        gameImage=gameImage,
                                        numberOfSteps=self.action_count,
                                        currentTouchPrediction=touchPrediction,
                                        wallInFront=inFront,
                                        didTouch=didTouch,
                                        turnLeftAndTouchPrediction=turnLeftAndTouchPrediction,
                                        wallOnLeft=onLeft,
                                        turnRightAndTouchPrediction=turnRightAndtouchPrediction,
                                        touchBehindPrediction=touchBehindPrediction,
                                        wallBehind=isBehind,
                                        touchAdjacentPrediction=isWallAdjacentPrediction,
                                        wallAdjacent=wallAdjacent,
                                        wallOnRight=onRight,
                                        distanceToAdjacent=distanceToAdjacent,
                                        distanceToAdjacentPrediction=distanceToAdjacentPrediction,
                                        distanceToLeft=distanceLeft,
                                        distanceToLeftPrediction=distanceLeftPrediction,
                                        distanceToRight=distanceRight,
                                        distanceToRightPrediction=distanceRightPrediction,
                                        distanceBack=distanceBack,
                                        distanceBackPrediction=distanceBackPrediction,
                                        wallLeftForward=wallLeftForward,
                                        wallLeftForwardPrediction=wallLeftForwardPrediction
                                        )
                    # time.sleep(1.0)

    def learn_from_behavior_policy_action(self):
        """Using the behaviour policy, selects an action. After selecting an action, updates the GVFs based on the
        action."""
        # todo: this is set as a variable in learn_from_action; we don't need to have two dependent calls...
        action = self.behavior_policy.mostly_forward_and_touch_policy(self.state)
        self.learn_from_action(action)

    def learn_from_action(self, action):
        """Given the most recent action, """
        self.action = action
        self.action_count += 1
        # If we've done 100 steps; pretty print the progress.
        if self.action_count % 100 == 0:
            print("Step " + str(self.action_count) + " ... ")

        self.old_state = self.state
        self.old_phi = self.phi

        observation = self.grid_world.takeAction(self.action)
        self.state = observation

        if self.old_state:
            yaw = self.old_state['yaw']
            xPos = self.old_state['x']
            zPos = self.old_state['y']

        yaw = self.state['yaw']
        xPos = self.state['x']
        zPos = self.state['y']

        self.phi = self.state_representation.get_phi(previous_phi=self.old_phi, state=self.state,
                                                     previous_action=self.action)

        # Do the learning
        self.learn()

        # Update our display (for debugging and progress reporting)
        self.update_ui()

    def start(self):
        """Initializes the plotter and runs the experiment."""
        self.action = self.behavior_policy.ACTIONS['extend_hand']  # the first action is always 'touch'
        # Loop until mission ends:
        while self.action_count < self.steps_before_prompting_for_action:
            # Select and send action. Need to sleep to give time for simulator to respond
            self.learn_from_behavior_policy_action()
        self.display.root.mainloop()
        print("Mission ended")
        # Mission has ended.


# fg.read_gvf_weights()
fg = Foreground(show_display=True, steps_before_updating_display=0, steps_before_prompting_for_action=12)

fg.start()
