""""""
from random import randint
import random


class BehaviorPolicy:
    """Specifies the behaviour policy for an agent to follow"""

    def __init__(self):

        self.lastAction = 0
        self.number_of_steps = 0
        self.ACTIONS = {
            'forward': "forward",
            'turn_left': "turn_left",
            'turn_right': "turn_right",
            'extend_hand': "extend_hand"
        }

    def policy(self, state):
        """"""
        self.number_of_steps += 1
        is_facing_wall = state[-1:] == 1  # Last bit in the feature representation represents facing the wall
        if is_facing_wall:                      # if the agent is facing a wall...
            return self.ACTIONS['look_left']    # look left ...
        else:
            return self.ACTIONS['forward']      # ... otherwise move forwards

    def random_turn_policy(self, state):
        """Randomly selects between turning left and turning right."""
        moves = [self.ACTIONS['turn_left'], self.ACTIONS['turn_right']]
        return moves[randint(0, 1)]

    def forward_then_left_policy(self, state):
        """Every 20th step, the agent turns left; otherwise the agent moves forwards."""
        self.number_of_steps += 1
        if self.number_of_steps % 20 == 0:
            return self.turnLeftPolicy(state)
        else:
            return self.moveForwardPolicy(state)

    def mostly_forward_policy(self, state):
        """Every 21st step, the policy chooses a random action; otherwise it moves forwards."""
        if self.number_of_steps % 21 == 0:
            return self.randomPolicy(state)
        else:
            return self.moveForwardPolicy(state)

    def mostly_forward_and_touch_policy(self, state):
        """Incr"""
        self.number_of_steps += 1

        if self.number_of_steps % 50 == 0:
            return self.turnRightPolicy(state)
        elif (self.number_of_steps - 1) % 50 == 0:
            return self.turnRightPolicy(state)
        elif ((self.number_of_steps - 2) % 50) == 0:
            return self.turnRightPolicy(state)
        elif self.number_of_steps % 7 == 0:
            return self.random_turn_policy(state)
        elif self.number_of_steps % 8 == 0:
            return self.random_turn_policy(state)
        elif self.number_of_steps % 19 == 0:
            return self.randomPolicy(state)
        elif self.number_of_steps % 21 == 0:
            return self.mostlyForwardPolicy(state)
        elif self.number_of_steps % 23 == 0:
            return self.mostly_forward_policy(state)
        else:
            if self.number_of_steps % 2 == 0 and self.number_of_steps < 30000:
                return self.ACTIONS['extend_hand']
            elif (self.number_of_steps - 1) % 4 == 0:
                return self.randomPolicy(state)
            else:
                return self.mostly_forward_policy(state)

    def extendHandPolicy(self, state):
        """For any state, extends the hand"""
        return self.ACTIONS['extend_hand']

    def randomPolicy(self, state):
        """For any state, randomly chooses action"""
        return self.ACTIONS[random.choice(list(self.ACTIONS.keys()))]

    def moveForwardPolicy(self, state):
        """For any state, moves forwards"""
        return self.ACTIONS['forward']

    def turnLeftPolicy(self, state):
        """For any state, turns left"""
        return self.ACTIONS['turn_left']

    def turnRightPolicy(self, state):
        """For any state, turns right"""
        return self.ACTIONS['turn_right']

    def epsilonGreedyPolicy(self, state):
        return NotImplementedError("This policy has not been implemented.")