import h5py
import numpy as np
from highdimsphadia import hiPhaData


class splitPhaSpace(hiPhaData):
    def __init__(self):
        super().__init__()
        self.n_t = None
        self.n_x = None
        self.min_number_of_data = None
        self.phase_kinds = None
        self.directions = [
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [-1, 0, 0, 0]
        ]

    def is_point_legal(self, point):
        if point[0] < 0 or point[0] > self.n_t - 0.6:
            return False
        if sum(point[1:]) > self.n_x - 0.6:
            return False
        if any(p < 0 for p in point[1:]):
            return False
        return True

    def hash_ergodic_array(self):
        self.n_x = int(1 / self.x_increment + 0.4) + 1
        self.n_t = int(1 / self.t_increment + 0.4) + 1
        hash_array = [[[[0] * self.n_x for _ in range(self.n_x)] for _ in range(self.n_x)] for _ in range(self.n_t)]
        return hash_array

    def create_phase_hash_list(self, points_for_hash):
        phase_hash_array = self.hash_ergodic_array()

        for i in range(points_for_hash.shape[0]):
            phase_hash_array[int(points_for_hash[i, 0] + 0.4)][int(points_for_hash[i, 1] + 0.4)][
                int(points_for_hash[i, 2] + 0.4)][int(points_for_hash[i, 3] + 0.4)] = 1

        return phase_hash_array

    def is_valid(self, point, visited, phase_hash_list):
        x, y, z, w = point
        if self.is_point_legal(point):
            if visited[x][y][z][w] == 0 and phase_hash_list[x][y][z][w] == 1:
                return True
        return False

    def dfs(self, point, phase_hash_list, visited, unvisited):
        stack = [point]
        count = 0
        while stack:
            current = stack.pop()
            x, y, z, w = current
            if visited[x][y][z][w] == 0:
                visited[x][y][z][w] = 1
                unvisited[x][y][z][w] = 0
                count += 1
                for d in self.directions:
                    new_point = [current[i] + d[i] for i in range(4)]
                    if self.is_valid(new_point, visited, phase_hash_list):
                        stack.append(new_point)

        return count

    def search_continuous_area_from_point(self, point, phase_hash_list, unvisited):
        visited = [[[[0 for _ in range(self.n_x)] for _ in range(self.n_x)] for _ in range(self.n_x)] for _ in
                   range(self.n_t)]
        count = self.dfs(point, phase_hash_list, visited, unvisited)
        return visited, unvisited, count

    def load_phase_data(self, path):
        f = h5py.File(path, 'r')
        phase_points = f['features'][()]
        f.close()

        # phase_points_for_hash = phase_points * 100
        phase_hash_list = self.create_phase_hash_list(phase_points)

        return phase_hash_list, phase_points.shape[0]

    def each_phase_area(self, phase_hash_list, number_of_points):
        unvisited = phase_hash_list.copy()
        each_phase_area_list = []

        def unvisited_point(unvisited):
            for i in range(self.n_t):
                for j in range(self.n_x):
                    for k in range(self.n_x):
                        for q in range(self.n_x):
                            if unvisited[i][j][k][q] == 1:
                                return [i, j, k, q]

            return False

        phase_area_count = 0

        while True:
            start_point = unvisited_point(unvisited)
            if start_point and (number_of_points > 30000):
                phase_area, unvisited, count = self.search_continuous_area_from_point(start_point, phase_hash_list,
                                                                                      unvisited)
                number_of_points -= count
                print(count)
                if count >= self.min_number_of_data:
                    each_phase_area_list.append(phase_area)
                    phase_area_count += 1
                    print(phase_area_count)

            else:
                break

        return each_phase_area_list, unvisited, number_of_points

    def hash_to_parallel_list(self, hash_list):
        para_list = []
        for i in range(self.n_t):
            for j in range(self.n_x):
                for k in range(self.n_x):
                    for q in range(self.n_x):
                        if hash_list[i][j][k][q] == 1:
                            para_list.append([i, j, k, q])
                            # print([i, j, k, q])

        return para_list

    def point_around(self, point):
        point_around_list = []
        for d in self.directions:
            temp = [point[i] + d[i] for i in range(4)]
            if self.is_point_legal(temp):
                point_around_list.append(temp)

        return point_around_list

    def phase_area_with_list_search(self, phase_hash_list, rest_unvisited):
        searched_point = []
        phase_area = []
        start_point = rest_unvisited[0]
        phase_area.append(start_point)
        searched_point.append(start_point)
        rest_unvisited.pop(rest_unvisited.index(start_point))
        point_around_list = self.point_around(start_point)
        while point_around_list:
            point_around_list_temp = []
            for i in point_around_list:
                if phase_hash_list[i[0]][i[1]][i[2]][i[3]] == 1:
                    phase_area.append(i)
                    searched_point.append(i)
                    rest_unvisited.pop(rest_unvisited.index(i))
                    point_around_i = self.point_around(i)
                    for j in point_around_i:
                        if not j in searched_point and (not j in point_around_list_temp):
                            point_around_list_temp.append(j)
                            print(j)

                else:
                    searched_point.append(i)

            point_around_list = point_around_list_temp

        return phase_area, rest_unvisited

    def rest_phase_area_list(self, rest_unvisited_points_list, phase_hash_list):
        rest_phase_area = []
        while rest_unvisited_points_list:
            each_rest_phase_area, rest_unvisited_points_list = self.phase_area_with_list_search(phase_hash_list,
                                                                                                rest_unvisited_points_list)
            if len(each_rest_phase_area) >= self.min_number_of_data:
                rest_phase_area.append(each_rest_phase_area)
                print(len(each_rest_phase_area))

        return rest_phase_area

    def all_phase_area_in_one_kind(self, path):
        all_phase_area_list = []
        phase_hash_list, number_of_points = self.load_phase_data(path)
        if number_of_points >= self.min_number_of_data:
            print(path[45:])
            each_phase_area_list, rest_unvisited_points, rest_number_of_points = self.each_phase_area(phase_hash_list,
                                                                                                      number_of_points)
            for i in each_phase_area_list:
                all_phase_area_list.append(self.hash_to_parallel_list(i))

            # print(len(each_phase_area_list))
            rest_unvisited_points = self.hash_to_parallel_list(rest_unvisited_points)
            rest_phase_area = self.rest_phase_area_list(rest_unvisited_points, phase_hash_list)

            for i in rest_phase_area:
                all_phase_area_list.append(i)

        # print(len(rest_phase_area))

        return all_phase_area_list

    def integrate_search(self, phase_data_path, save_split_data_path, min_number_of_data):
        self.min_number_of_data = min_number_of_data

        for i in range(self.phase_kinds.shape[0]):
            all_phase_area_list = self.all_phase_area_in_one_kind(
                phase_data_path + '\\' + self.phase_kinds[
                    i, 0] + '.h5')
            divided_phases = np.array([-1, -1, -1, -1])
            divided_phases = np.expand_dims(divided_phases, axis=0)
            if all_phase_area_list:
                for phase_areas in all_phase_area_list:
                    temp = np.array(phase_areas)
                    divided_phases = np.append(divided_phases, temp, axis=0)
                    temp = np.array([-1, -1, -1, -1])
                    temp = np.expand_dims(temp, axis=0)
                    divided_phases = np.append(divided_phases, temp, axis=0)

                f = h5py.File(
                    save_split_data_path + '\\' +
                    self.phase_kinds[i, 0] + '.h5', 'w')
                f.create_dataset('features', data=divided_phases)
                f.close()


FeCrNiMn = splitPhaSpace()
FeCrNiMn.phase_kinds = np.load('phase kinds in FeCrNiMn.npy')
FeCrNiMn.x_increment = 0.01
FeCrNiMn.t_increment = 0.005
FeCrNiMn.integrate_search('D:\\', 'D:\\', 200)
