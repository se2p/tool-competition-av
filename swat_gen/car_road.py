#
import random as rm
import numpy as np
import math as m
import matplotlib.pyplot as plt
import json


class Map:
    '''Class that conducts transformations to vectors automatically,
    using the commads "go straight", "turn left", "turn right".
    As a result it produces a set of points corresponding to a road
    '''
    def __init__(self, map_size, init=0, a="", b=""):
        self.map_size = map_size
        self.width = 10
        self.max_x = map_size
        self.max_y = map_size
        self.min_x = 0
        self.min_y = 0
        self.radius = 25
        if init == 0:
            self.init_pos, self.init_end = self.init_pos()
        else:
            self.init_pos, self.init_end = a, b

        self.current_pos = [self.init_pos, self.init_end]
        self.all_position_list = [[self.init_pos, self.init_end]]

    def init_pos(self):
        '''select a random initial position from the middle of 
        one of the boundaries
        '''
        option = rm.randint(0, 3)
        if option == 0:
            pos = np.array((self.max_x / 2, 0))
            end = np.array((pos[0] + self.width, pos[1]))
        elif option == 1:
            pos = np.array((0, self.max_y / 2))
            end = np.array((pos[0], pos[1] - self.width))
        elif option == 2:
            pos = np.array((self.max_x / 2, self.max_y))
            end = np.array((pos[0] + self.width, pos[1]))
        elif option == 3:
            pos = np.array((self.max_x, self.max_y / 2))
            end = np.array((pos[0], pos[1] + self.width))

        return pos, end

    def point_in_range(self, a):
        '''check if point is in the acceptable range
        '''
        if 0 <= a[0] <= self.max_x and 0 <= a[1] <= self.max_y:
            return 1
        else:
            return 0

    def go_straight(self, distance):
        '''transform a vector paralelly to the previous one
        '''
        a = self.current_pos[0]
        b = self.current_pos[1]

        if self.point_in_range(a) == 0 or self.point_in_range(b) == 0:
            # print("Point not in range...")
            return False

        if (b - a)[1] > 0:
            p_a = b
            p_b = a
        elif (b - a)[1] < 0:
            p_a = a
            p_b = b
        elif (b - a)[1] == 0:
            if (b - a)[0] > 0:
                p_a = b
                p_b = a
            else:
                p_a = a
                p_b = b

        vec_ang = self.get_angle(p_a, p_b)

        sector = self.get_sector()

        u_v = (p_a - p_b) / np.linalg.norm(p_b - p_a)

        if sector == 0:
            if 0 <= vec_ang <= 90:
                R = np.array([[0, -1], [1, 0]])  # turn anti-clockwise
            elif 90 < vec_ang <= 180:
                if self.prev_from_top() == 0:
                    R = np.array([[0, -1], [1, 0]])  # anti-clockwise
                elif self.prev_from_top() == 1:
                    R = np.array([[0, 1], [-1, 0]])  #  clockwise
            else:
                print("Invalid angle")
        elif sector == 1:
            if 0 <= vec_ang <= 180:
                R = np.array([[0, 1], [-1, 0]])  # turn clockwise
            else:
                print("Invalid angle")
        elif sector == 2:
            if 0 <= vec_ang <= 90:
                R = np.array([[0, 1], [-1, 0]])  # turn clockwise
            elif 90 < vec_ang <= 180:
                if self.prev_is_left() == 1 and self.prev_from_top() == 0:
                    R = np.array([[0, 1], [-1, 0]])  # turn clockwise
                elif self.prev_is_left() == 1 and self.prev_from_top() == 1:
                    R = np.array([[0, -1], [1, 0]])
                elif self.prev_is_left() == 0:
                    R = np.array([[0, -1], [1, 0]])  # turn clockwise
            else:
                print("Invalid angle")
        elif sector == 3:
            if 0 <= vec_ang <= 90:
                is_top = self.prev_from_top()
                if is_top == 1:
                    R = np.array([[0, 1], [-1, 0]])
                if is_top == 0:
                    R = np.array([[0, -1], [1, 0]])
            elif 90 < vec_ang <= 180:
                prev_below = self.prev_from_top()
                if prev_below == 1:
                    R = np.array([[0, 1], [-1, 0]])
                if prev_below == 0:
                    R = np.array([[0, -1], [1, 0]])
            else:
                print("Invalid angle")

        u_v = R.dot(u_v)
        p_a_ = p_a + u_v * distance
        p_b_ = p_b + u_v * distance
        self.current_pos = [p_a_, p_b_]
        self.all_position_list.append(self.current_pos)
        return True

    def clockwise_turn_top(self, angle, p_a, p_b):
        angle += 180
        radius = self.radius

        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_a + u_v * radius

        o_b_norm = np.linalg.norm(o_o - p_b)

        o_a_norm = np.linalg.norm(o_o - p_a)

        o_b = (o_o - p_b) / o_b_norm
        o_a = (o_o - p_a) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), np.sin(m.radians(angle))],
                [-np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )
        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm

        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def clockwise_turn_bot(self, angle, p_a, p_b):
        radius = self.radius
        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_b - u_v * radius
        o_b_norm = np.linalg.norm(o_o - p_b)
        o_a_norm = np.linalg.norm(o_o - p_a)
        o_b = (p_b - o_o) / o_b_norm
        o_a = (p_a - o_o) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), np.sin(m.radians(angle))],
                [-np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )

        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm
        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def anticlockwise_turn_top(self, angle, p_a, p_b):
        angle += 180
        radius = self.radius
        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_a + u_v * radius

        o_b_norm = np.linalg.norm(o_o - p_b)

        o_a_norm = np.linalg.norm(o_o - p_a)

        o_b = (o_o - p_b) / o_b_norm
        o_a = (o_o - p_a) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), -np.sin(m.radians(angle))],
                [np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )
        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm

        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def anticlockwise_turn_bot(self, angle, p_a, p_b):
        radius = self.radius
        u_v = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        o_o = p_b - u_v * radius

        o_b_norm = np.linalg.norm(o_o - p_b)
        o_a_norm = np.linalg.norm(o_o - p_a)
        o_b = (p_b - o_o) / o_b_norm
        o_a = (p_a - o_o) / o_a_norm

        R = np.array(
            [
                [np.cos(m.radians(angle)), -np.sin(m.radians(angle))],
                [np.sin(m.radians(angle)), np.cos(m.radians(angle))],
            ]
        )
        o_b_ = R.dot(o_b) * o_b_norm
        o_a_ = R.dot(o_a) * o_a_norm

        p_a_ = o_o + o_a_
        p_b_ = o_o + o_b_

        return [p_a_, p_b_]

    def turn_right(self, angle):
        a = self.current_pos[0]
        b = self.current_pos[1]
        if self.point_in_range(a) == 0 or self.point_in_range(b) == 0:
            # print("Point not in range...")
            return False

        # p_a will be the end of vector, p_b - the start

        if (b - a)[1] > 0:
            p_a = b
            p_b = a
        elif (b - a)[1] < 0:
            p_a = a
            p_b = b
        elif (b - a)[1] == 0:
            if (b - a)[0] > 0:
                p_a = b
                p_b = a
            else:
                p_a = a
                p_b = b

        vec_ang = self.get_angle(p_a, p_b)
        sector = self.get_sector()

        if sector == 0:
            if 0 <= vec_ang <= 90:
                self.current_pos = self.clockwise_turn_top(angle, p_a, p_b)
            elif 90 < vec_ang <= 180:
                self.current_pos = self.clockwise_turn_bot(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)
        elif sector == 1:
            if 0 <= vec_ang <= 180:
                self.current_pos = self.clockwise_turn_bot(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)
        elif sector == 2:
            if 0 <= vec_ang <= 90:
                self.current_pos = self.clockwise_turn_bot(angle, p_a, p_b)
            elif 90 < vec_ang <= 180:
                if self.prev_is_left() == 0:
                    self.current_pos = self.clockwise_turn_top(angle, p_a, p_b)
                elif self.prev_is_left() == 1 and self.prev_from_top() == 0:
                    self.current_pos = self.clockwise_turn_bot(angle, p_a, p_b)
                elif self.prev_is_left() == 1 and self.prev_from_top() == 1:
                    self.current_pos = self.clockwise_turn_top(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)
        elif sector == 3:
            if 0 <= vec_ang <= 180:
                self.current_pos = self.clockwise_turn_top(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)

        return True

    def turn_left(self, angle):

        a = self.current_pos[0]
        b = self.current_pos[1]

        if self.point_in_range(a) == 0 or self.point_in_range(b) == 0:
            # print("Point not in range...")
            return False

        # p_a will be the end of vector, p_b - the start

        if (b - a)[1] > 0:
            p_a = b
            p_b = a
        elif (b - a)[1] < 0:
            p_a = a
            p_b = b
        elif (b - a)[1] == 0:
            if (b - a)[0] > 0:
                p_a = b
                p_b = a
            else:
                p_a = a
                p_b = b

        vec_ang = self.get_angle(p_a, p_b)
        sector = self.get_sector()

        if sector == 0:
            if 0 <= vec_ang <= 90:
                self.current_pos = self.anticlockwise_turn_bot(angle, p_a, p_b)
            elif 90 < vec_ang <= 180:
                self.current_pos = self.anticlockwise_turn_top(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)
        elif sector == 1:
            if 0 <= vec_ang <= 90:
                if self.prev_is_below() == 1:
                    self.current_pos = self.anticlockwise_turn_bot(angle, p_a, p_b)
                elif self.prev_is_below() == 0:
                    self.current_pos = self.anticlockwise_turn_top(angle, p_a, p_b)
            if 90 < vec_ang <= 180:
                prev_top = self.prev_from_top()
                if prev_top == 0:  #
                    self.current_pos = self.anticlockwise_turn_top(angle, p_a, p_b)
                elif prev_top == 1:
                    self.current_pos = self.anticlockwise_turn_bot(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)
        elif sector == 2:
            if 0 <= vec_ang <= 90:
                self.current_pos = self.anticlockwise_turn_top(angle, p_a, p_b)
            elif 90 < vec_ang <= 180:
                if self.prev_is_left() == 0:
                    self.current_pos = self.anticlockwise_turn_bot(angle, p_a, p_b)
                elif self.prev_is_left() == 1 and self.prev_from_top() == 0:
                    self.current_pos = self.anticlockwise_turn_top(angle, p_a, p_b)
                elif self.prev_is_left() == 1 and self.prev_from_top() == 1:
                    self.current_pos = self.anticlockwise_turn_bot(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)
        elif sector == 3:
            if 0 <= vec_ang <= 180:
                self.current_pos = self.anticlockwise_turn_bot(angle, p_a, p_b)
            else:
                print("Invalid angle")
            self.all_position_list.append(self.current_pos)

        return True

    def prev_from_top(self):
        '''returns one if the previous vector
        comes to the top of the current vector and 
        0 if it comes to the bottom
        '''
        if len(self.all_position_list) == 1:
            if self.get_sector() == 3:
                return 0
            else:
                return 1
        elif len(self.all_position_list) > 1:
            last = self.all_position_list[-1]
            prev = self.all_position_list[-2]
            if last[1][1] > last[0][1]:
                last_max_y = last[1]
                last_min_y = last[0]
            else:
                last_max_y = last[0]
                last_min_y = last[1]

            if prev[1][1] > prev[0][1]:
                prev_max_y = prev[1]
                prev_min_y = prev[0]

            else:
                prev_max_y = prev[0]
                prev_min_y = prev[1]

            m_max = -(last_max_y[1] - prev_max_y[1]) / (last_max_y[0] - prev_max_y[0])
            m_min = -(last_min_y[1] - prev_min_y[1]) / (last_min_y[0] - prev_min_y[0])
            if abs(m_max - m_min) < 0.001:  # k are the same
                return 0
            else:
                return 1  # k are different

    def prev_is_left(self):
        '''returns one if previous vector is
        to the left of the current one'''
        if len(self.all_position_list) == 1:
            return 0
        elif len(self.all_position_list) > 1:
            last = self.all_position_list[-1]
            prev = self.all_position_list[-2]

            last_mid = (last[0] + last[1]) / 2
            prev_mid = (prev[0] + prev[1]) / 2

            default = 0

            if last_mid[0] > prev_mid[0]:
                return 1
            elif last_mid[0] < prev_mid[0]:
                return 0
            elif last_mid[0] == prev_mid[0]:
                if self.get_sector() == 0:
                    return 0
                elif self.get_sector() == 2:
                    return 0
                else:
                    return default

    def prev_is_below(self):
        '''returns one if previous vector is below the 
        current one
        '''
        if len(self.all_position_list) == 1:
            if self.get_sector() == 0:
                return 1
            elif self.get_sector() == 2:
                return 0
            elif self.get_sector() == 3:
                return 1
        elif len(self.all_position_list) > 1:
            last = self.all_position_list[-1]
            prev = self.all_position_list[-2]

            last_mid = (last[0] + last[1]) / 2
            prev_mid = (prev[0] + prev[1]) / 2

            default = 1

            if last_mid[1] > prev_mid[1]:
                return 1
            elif last_mid[1] < prev_mid[1]:
                return 0
            elif last_mid[1] == prev_mid[1]:
                return default

    def get_angle(self, p_a, p_b):
        '''returns angle between two vectors
        '''
        unit = np.array([1, 0])

        vec = (p_a - p_b) / np.linalg.norm(p_a - p_b)
        dot = np.dot(vec, unit)
        cos = dot
        angle = m.degrees(np.arccos(cos))
        return angle

    def get_sector(self):
        '''by using previous vector position draws a line 
        between current vector middle and previous vector middle,
        to see what boundary the line intersects, the sectors.
        It allows to get more info about direction of vector movement
        '''
        if len(self.all_position_list) == 1:
            last = self.all_position_list[-1]
            if last[0][1] == 0:
                return 0
            elif last[0][0] == 0:
                return 1
            elif last[0][1] == self.max_y:
                return 2
            elif last[0][0] == self.max_x:
                return 3
        elif len(self.all_position_list) > 1:
            last = self.all_position_list[-1]
            prev = self.all_position_list[-2]

            last_mid = (last[0] + last[1]) / 2
            prev_mid = (prev[0] + prev[1]) / 2

            if last_mid[0] - prev_mid[0] == 0:
                x_0 = last_mid[0]
                y_0 = 0

                x_2 = last_mid[0]
                y_2 = self.max_y

                sector_dict = {}
                sector_dict["0"] = np.array([x_0, y_0])
                sector_dict["2"] = np.array([x_2, y_2])
                result = []
                for sect in sector_dict:
                    point = sector_dict[sect]

                    prev2p = np.linalg.norm(point - prev_mid)
                    last2p = np.linalg.norm(prev_mid - last_mid)
                    tot = np.linalg.norm(last_mid - point)

                    if tot - 0.01 <= prev2p + last2p <= tot + 0.01:
                        result.append(int(sect))

            else:
                m = (last_mid[1] - prev_mid[1]) / (last_mid[0] - prev_mid[0])
                b = last_mid[1] - m * last_mid[0]

                sector_dict = {}
                if m == 0:
                    x_0 = -1
                else:
                    x_0 = -b / m
                y_0 = 0
                if 0 <= x_0 <= self.max_x:
                    sector_dict["0"] = np.array([x_0, y_0])

                x_1 = 0
                y_1 = b
                if 0 <= y_1 <= self.max_y:
                    sector_dict["1"] = np.array([x_1, y_1])
                if m == 0:
                    x_2 = -1
                else:
                    x_2 = (self.max_y - b) / m
                y_2 = self.max_y
                if 0 <= x_2 <= self.max_x:
                    sector_dict["2"] = np.array([x_2, y_2])

                x_3 = self.max_x
                y_3 = m * x_3 + b
                if 0 <= y_3 <= self.max_y:
                    sector_dict["3"] = np.array([x_3, y_3])
                result = []

                for sect in sector_dict:
                    point = sector_dict[sect]
                    prev2p = np.linalg.norm(point - prev_mid)
                    last2p = np.linalg.norm(prev_mid - last_mid)
                    tot = np.linalg.norm(last_mid - point)

                    if tot - 0.01 <= prev2p + last2p <= tot + 0.01:
                        result.append(int(sect))
        #print("sector", result)

        return result[0]
