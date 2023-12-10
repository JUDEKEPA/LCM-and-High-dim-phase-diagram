import h5py
import numpy as np
from conreaction import eutecticReaction, eutectoidReaction, precipitation


class alloyDatabase:
    def __init__(self, path=None):
        self.path = path
        self.phase_area_kinds_combined = []
        self.phase_area_kinds = None

    def import_phase_area_kinds(self, path):
        phase_area_kinds_ndarray = np.load(path)
        for i in range(phase_area_kinds_ndarray.shape[0]):
            self.phase_area_kinds_combined.append(phase_area_kinds_ndarray[i, 0])
        phase_area_kinds_list = []
        for i in range(phase_area_kinds_ndarray.shape[0]):
            phase_area_kinds_temp = phase_area_kinds_ndarray[i, 0].split('+')
            phase_area_kinds_list.append(phase_area_kinds_temp)

        self.phase_area_kinds = phase_area_kinds_list


    def eutectic_reaction(self, tolerance=2):
        eutectic_ra = eutecticReaction(self.path)
        eutectic_ra.liquid_lower_boundary()
        eutectic_ra.find_possible_eutectic_reaction(self.phase_area_kinds)
        eutectic_ra.possible_eutectic_phases_area_hash(self.phase_area_kinds_combined)
        eutectic_ra.find_eutectic_contents(tolerance)

        return eutectic_ra

    def ternary_eutectic_reaction(self, tolerance=1):
        eutectic_ra = eutecticReaction(self.path)
        eutectic_ra.liquid_lower_boundary()
        eutectic_ra.find_possible_ternary_eutectic_reaction(self.phase_area_kinds)
        eutectic_ra.possible_ternary_eutectic_phases_area_hash(self.phase_area_kinds_combined)
        eutectic_ra.find_ternary_eutectic_contents(tolerance)

        return eutectic_ra

    def eutectoid_reaction(self):
        eutectoid = eutectoidReaction(self.path)
        a = eutectoid.single_double_phases(self.phase_area_kinds)
        b = eutectoid.possible_eutectoid_phases_area_hash(self.phase_area_kinds_combined)
        c = eutectoid.all_eutectoid_reaction()
        return c

    def precipitation_reaction(self):
        precipitation_ra = precipitation()
        file_names = precipitation_ra.possible_precipitation(self.phase_area_kinds)
        #file_names = precipitation_ra.precipitation_area_hash(self.path, self.phase_area_kinds_combined)

        return file_names


FeCrNiMn_alloy_design = alloyDatabase('..\\basecode\\each phase kind data\\each kind with divided area\\')
FeCrNiMn_alloy_design.import_phase_area_kinds('..\\basecode\\phase kinds in FeCrNiMn.npy')
eutectic_FeNiCrMn = FeCrNiMn_alloy_design.eutectic_reaction(1)
a = eutectic_FeNiCrMn.t_extent_condition(5)
eutectic_FeNiCrMn.draw_3D_fig(a)

#a = FeCrNiMn_alloy_design.eutectoid_reaction()

#a = FeCrNiMn_alloy_design.precipitation_reaction()

