from Voronoi import *


class GridWorld:

    def __init__(self, model_file, initial_x=0, initial_y=0, initial_yaw=0):
        # Initialize the environment using the configuration file
        print("Initializing grid world ...")
        self.grids = {}
        with open(model_file, 'rb') as gridDataFile:
            self.grids = pickle.load(gridDataFile)

        self.current_x = initial_x
        self.current_y = initial_y
        self.current_yaw = initial_yaw
        self.ACTIONS = [
            'forward',
            'turn_left',
            'turn_right',
            'extend_hand'
        ]

    def grid_for(self, x, y):
        k = self.key_name_for(x, y)
        if k in self.grids:
            return self.grids[k]
        else:
            return None

    @staticmethod
    def key_name_for(x, y):
        return str(x) + ',' + str(y)

    def take_action(self, action):

        # Set the new orientation
        if action > 3:
            print("Error. specified action not in action set")
            return

        did_touch = False
        if action == 1:
            self.current_yaw = (self.current_yaw - 90) % 360
        elif action == 2:
            self.current_yaw = (self.current_yaw + 90) % 360
        elif action == 3:
            desired_key = ''
            if self.current_yaw == 0:
                desired_key = self.key_name_for(self.current_x, self.current_y + 1)
            elif self.current_yaw == 90:
                desired_key = self.key_name_for(self.current_x - 1, self.current_y)
            elif self.current_yaw == 180:
                desired_key = self.key_name_for(self.current_x, self.current_y - 1)
            elif self.current_yaw == 270:
                desired_key = self.key_name_for(self.current_x + 1, self.current_y)
            if desired_key not in self.grids:
                did_touch = True
        elif action == 4:
            if self.current_yaw == 0:
                desired_key = self.key_name_for(self.current_x, self.current_y + 1)
                if desired_key in self.grids:
                    # Move south
                    self.current_y = self.current_y + 1
            elif self.current_yaw == 90:
                desired_key = self.key_name_for(self.current_x - 1, self.current_y)
                if desired_key in self.grids:
                    # Move west
                    self.current_x = self.current_x - 1
            elif self.current_yaw == 180:
                desired_key = self.key_name_for(self.current_x, self.current_y - 1)
                if desired_key in self.grids:
                    # move north
                    self.current_y = self.current_y - 1
            elif self.current_yaw == 270:
                desired_key = self.key_name_for(self.current_x + 1, self.current_y)
                if desired_key in self.grids:
                    # move east
                    self.current_x = self.current_x + 1

        current_grid_key = self.key_name_for(self.current_x, self.current_y)

        pixel_data = self.grids[current_grid_key][str(self.current_yaw)]
        return {'visionData': pixel_data, 'touchData': did_touch, 'reward': 0, 'x': self.current_x, 'y': self.current_y,
                'yaw': self.current_yaw}
