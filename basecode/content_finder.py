import sys
sys.path.append('../../')
import numpy as np
from alloydesign.commall import alloyDatabase
from alloydesign.conreaction import eutecticReaction
import copy
import h5py
from io import BytesIO
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class eutecticContentFinder(alloyDatabase):
    def __init__(self):
        super().__init__()
        self.eutectic_class = None
        self.path = '.\each phase kind data\each kind with divided area\\'
        self.get_eutectic_class()


    def get_eutectic_class(self, tolerance=2):
        self.import_phase_area_kinds('.\divided phase kinds in FeCrNiMn.npy')
        self.eutectic_class = self.eutectic_reaction(tolerance)

    def get_certain_eutectic_list(self, eutectic_kind, t_extent=5, tolerance=2):
        eutectic_contents = self.eutectic_class.find_eutectic_contents(tolerance)
        eutectic_contents_fit_temper_con = self.eutectic_class.t_extent_condition(eutectic_contents, t_extent)
        certain_eutectic_list = []
        for i in eutectic_contents_fit_temper_con:
            if i[1] == eutectic_kind:
                certain_eutectic_list.append(i)

        return certain_eutectic_list

    def get_3D_image(self, eutectic_kind, t_extent=5, tolerance=2):
        certain_eutectic_list = self.get_certain_eutectic_list(eutectic_kind, t_extent, tolerance)
        img = eutecticReaction.draw_3D_fig(certain_eutectic_list, eutectic_kind)
        return img


class eutectoidContentFinder:
    def __init__(self):
        self.path = '.\each phase kind data\each kind with divided area\\'
        self.phase_area_kinds = None
        self.phase_area_kinds_combined = []

    def import_phase_area_kinds(self, path):
        phase_area_kinds_ndarray = np.load(path)
        for i in range(phase_area_kinds_ndarray.shape[0]):
            self.phase_area_kinds_combined.append(phase_area_kinds_ndarray[i, 0])
        phase_area_kinds_list = []
        for i in range(phase_area_kinds_ndarray.shape[0]):
            phase_area_kinds_temp = phase_area_kinds_ndarray[i, 0].split('+')
            phase_area_kinds_list.append(phase_area_kinds_temp)

        self.phase_area_kinds = phase_area_kinds_list

    def get_certain_eutectoid_contents(self, single_phase_name, double_phase_name, lowest_t=5, highest_t=40, single_phase_t_extent=4, double_phase_t_extent=4):
        certain_eutectoid_contents_list = []
        conditional_certain_eutectoid_contents_list = []
        content_judge_hash = [[[0 for _ in range(101)] for _ in range(101)] for _ in range(101)]
        phase_area_hash = [[[[0 for _ in range(101)] for _ in range(101)] for _ in range(101)] for _ in range(101)]
        f = h5py.File(self.path + single_phase_name[0] + '.h5', 'r')
        single_phase_area = f['features'][()]
        f.close()
        for i in self.phase_area_kinds_combined:
            if i.count('+') == 1:
                if (double_phase_name[0] in i) and (double_phase_name[1] in i):
                    f = h5py.File(self.path + i + '.h5', 'r')
                    double_phase_area = f['features'][()]
                    f.close()

        for i in range(single_phase_area.shape[0]):
            if single_phase_area[i, 0] != 111:
                content_judge_hash[single_phase_area[i, 1]][single_phase_area[i, 2]][single_phase_area[i, 3]] = 1
                phase_area_hash[single_phase_area[i, 0]][single_phase_area[i, 1]][single_phase_area[i, 2]][single_phase_area[i, 3]] = single_phase_name
        for i in range(double_phase_area.shape[0]):
            if double_phase_area[i, 0] != 111:
                if content_judge_hash[double_phase_area[i, 1]][double_phase_area[i, 2]][double_phase_area[i, 3]] == 1 and (not [double_phase_area[i, 1], double_phase_area[i, 2], double_phase_area[i, 3]] in certain_eutectoid_contents_list):
                    certain_eutectoid_contents_list.append([double_phase_area[i, 1], double_phase_area[i, 2], double_phase_area[i, 3]])
                    phase_area_hash[double_phase_area[i, 0]][double_phase_area[i, 1]][double_phase_area[i, 2]][double_phase_area[i, 3]] = double_phase_name

        count_single = 0
        count_double = 0
        for contents in certain_eutectoid_contents_list:
            for temperature in range(lowest_t, highest_t+1):
                if phase_area_hash[temperature][contents[0]][contents[1]][contents[2]] == single_phase_name:
                    count_single += 1
                elif phase_area_hash[temperature][contents[0]][contents[1]][contents[2]] == double_phase_name:
                    count_double += 1

            if count_single >= single_phase_t_extent and count_double >= double_phase_t_extent:
                conditional_certain_eutectoid_contents_list.append(contents)

        return conditional_certain_eutectoid_contents_list

    def draw_eutectoid_3D_fig(self, single_phase_name, double_phase_name, lowest_t=5, highest_t=40, single_phase_t_extent=4, double_phase_t_extent=4):
        conditional_certain_eutectoid_contents_list = self.get_certain_eutectoid_contents(single_phase_name, double_phase_name, lowest_t, highest_t, single_phase_t_extent, double_phase_t_extent)
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        x, y, z = [], [], []
        for i in conditional_certain_eutectoid_contents_list:
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
        ax1.scatter3D(x, y, z, label=single_phase_name[0]+'->'+double_phase_name[0]+'+'+double_phase_name[1])
        ax1.set_xlabel('Fe, at. %')
        ax1.set_ylabel('Ni, at. %')
        ax1.set_zlabel('Cr, at. %')
        plt.legend()
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return img

class precipitationContentFinder:
    def __init__(self):
        self.precipitation_kinds_list = None
        self.path = '.\precipitation phase with volume\\'

    def import_precipitation_kind(self, kind_path):
        precipitation_kinds = np.load(kind_path)
        precipitation_kinds_list = []
        for i in range(precipitation_kinds.shape[0]):
            precipitation_kinds_list.append(precipitation_kinds[i, 0].split('-'))

        self.precipitation_kinds_list = precipitation_kinds_list

        return precipitation_kinds_list

    def find_con_precipitation_contents(self, matrix, precipitation, precipitation_min_vol=0, precipitation_max_vol=1):
        for precipitation_kinds in self.precipitation_kinds_list:
            if (matrix in precipitation_kinds[0]) and (matrix in precipitation_kinds[1]) and (precipitation in precipitation_kinds[1]):
                f = h5py.File(self.path + precipitation_kinds[0] + '-' + precipitation_kinds[1] + '.h5', 'r')
                precipitation_content_vol = f['features'][()]
                f.close()

        content_list = []

        for i in range(precipitation_content_vol.shape[0]):
            if precipitation_content_vol[i, 3] >= precipitation_min_vol and precipitation_content_vol[i, 3] <= precipitation_max_vol:
                content_list.append([precipitation_content_vol[i, 0], precipitation_content_vol[i, 1], precipitation_content_vol[i, 2]])

        return content_list

    def draw_precipitation_content_figure(self, matrix, precipitation, precipitation_min_vol, precipitation_max_vol):
        conditional_precipitation_contents_list = self.find_con_precipitation_contents(matrix, precipitation, precipitation_min_vol, precipitation_max_vol)
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        x, y, z = [], [], []
        for i in conditional_precipitation_contents_list:
            x.append(i[0])
            y.append(i[1])
            z.append(i[2])
        ax1.scatter3D(x, y, z, label=matrix+'->'+matrix+'+'+precipitation)
        ax1.set_xlabel('Fe, at. %')
        ax1.set_ylabel('Ni, at. %')
        ax1.set_zlabel('Cr, at. %')
        plt.legend()
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        #plt.close()

        return img
