import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from basecode.calcpha import calc_model, output_normalize
from basecode.generatephase import outputDataProcessor


class eutecticReaction:
    def __init__(self, path=None):
        self.ternary_eutectic_reaction_contents = None
        self.possible_ternary_phases_area = None
        self.possible_ternary_reaction = None
        self.eutectic_contents_fit_temper_con = None
        self.possible_eutectic_reaction = None
        self.path = path
        self.liquid_boundary_list = None
        self.possible_eutectic_phases_area = None
        self.eutectic_contents = None

    @staticmethod
    def find_phase_neighboring_liquid(phase_area_kind_list):
        phase_neighboring_liquid = []
        for each_phase_kind in phase_area_kind_list:
            if len(each_phase_kind) == 2:
                if 'LIQUID' in each_phase_kind:
                    for phase in each_phase_kind:
                        if phase != 'LIQUID':
                            phase_neighboring_liquid.append(phase)

        return phase_neighboring_liquid

    def find_possible_eutectic_reaction(self, phase_area_kind_list):
        possible_eutectic_reaction = []
        phase_neighboring_liquid = eutecticReaction.find_phase_neighboring_liquid(phase_area_kind_list)
        if len(phase_neighboring_liquid) < 2:
            return False
        left = 0
        right = 1
        while left < len(phase_neighboring_liquid) - 1:
            for i in phase_area_kind_list:
                if len(i) == 2:
                    if (phase_neighboring_liquid[left] in i) and (phase_neighboring_liquid[right] in i):
                        possible_eutectic_reaction.append(
                            [phase_neighboring_liquid[left], phase_neighboring_liquid[right]])
            right += 1
            if right == len(phase_neighboring_liquid):
                left += 1
                right = left + 1

        self.possible_eutectic_reaction = possible_eutectic_reaction

        return possible_eutectic_reaction

    def find_possible_ternary_eutectic_reaction(self, phase_area_kind_list):
        possible_ternary_reaction = []
        phase_neighboring_liquid = eutecticReaction.find_phase_neighboring_liquid(phase_area_kind_list)
        if len(phase_neighboring_liquid) < 2:
            return False

        for i in phase_area_kind_list:
            if len(i) == 3:
                possible_ternary = False
                for j in i:
                    if j in phase_neighboring_liquid:
                        possible_ternary = True
                    else:
                        possible_ternary = False
                        break
                if possible_ternary:
                    possible_ternary_reaction.append(i)

        self.possible_ternary_reaction = possible_ternary_reaction

        return possible_ternary_reaction

    def liquid_lower_boundary(self):
        liquid_boundary_list = []

        f = h5py.File(self.path + 'LIQUID.h5', 'r')
        liquid_area = f['features'][()]
        f.close()

        liquid_hash = [[[[0 for _ in range(101)] for _ in range(101)] for _ in range(101)] for _ in range(201)]

        for i in range(liquid_area.shape[0]):
            x, y, z, w = liquid_area[i, 0], liquid_area[i, 1], liquid_area[i, 2], liquid_area[i, 3]
            if x != -1:
                liquid_hash[x][y][z][w] = 1

        for i in range(101):
            for j in range(101):
                for k in range(101):
                    for q in range(201):
                        if liquid_hash[q][i][j][k] == 1:
                            liquid_boundary_list.append([q, i, j, k])
                            break
            print(i)

        self.liquid_boundary_list = liquid_boundary_list

        return liquid_boundary_list

    def possible_eutectic_phases_area_hash(self, phase_area_kinds_combined):
        possible_eutectic_phase_area = [[[[0 for _ in range(101)] for _ in range(101)] for _ in range(101)] for _ in
                                        range(201)]
        for possible_two_eutect_phase in self.possible_eutectic_reaction:
            if (possible_two_eutect_phase[0] + '+' + possible_two_eutect_phase[1]) in phase_area_kinds_combined:
                doc_name = possible_two_eutect_phase[0] + '+' + possible_two_eutect_phase[1]
            else:
                doc_name = possible_two_eutect_phase[1] + '+' + possible_two_eutect_phase[0]

            f = h5py.File(self.path + doc_name + '.h5', 'r')
            temp_area = f['features'][()]
            f.close()

            for i in range(temp_area.shape[0]):
                if temp_area[i, 0] != -1:
                    possible_eutectic_phase_area[temp_area[i, 0]][temp_area[i, 1]][temp_area[i, 2]][
                        temp_area[i, 3]] = doc_name

            print(doc_name)

        self.possible_eutectic_phases_area = possible_eutectic_phase_area

        return possible_eutectic_phase_area

    def possible_ternary_eutectic_phases_area_hash(self, phase_area_kinds_combined):
        doc_name = None
        possible_ternary_phase_area = [[[[0 for _ in range(101)] for _ in range(101)] for _ in range(101)] for _ in
                                       range(201)]
        for possible_three_eutect_phase in self.possible_ternary_reaction:
            if (possible_three_eutect_phase[0] + '+' + possible_three_eutect_phase[1] + '+' +
                possible_three_eutect_phase[2]) in phase_area_kinds_combined:
                doc_name = possible_three_eutect_phase[0] + '+' + possible_three_eutect_phase[1] + '+' + \
                           possible_three_eutect_phase[2]

            elif (possible_three_eutect_phase[0] + '+' + possible_three_eutect_phase[2] + '+' +
                  possible_three_eutect_phase[1]) in phase_area_kinds_combined:
                doc_name = possible_three_eutect_phase[0] + '+' + possible_three_eutect_phase[2] + '+' + \
                           possible_three_eutect_phase[1]

            elif (possible_three_eutect_phase[1] + '+' + possible_three_eutect_phase[0] + '+' +
                  possible_three_eutect_phase[2]) in phase_area_kinds_combined:
                doc_name = possible_three_eutect_phase[1] + '+' + possible_three_eutect_phase[0] + '+' + \
                           possible_three_eutect_phase[2]

            elif (possible_three_eutect_phase[1] + '+' + possible_three_eutect_phase[2] + '+' +
                  possible_three_eutect_phase[0]) in phase_area_kinds_combined:
                doc_name = possible_three_eutect_phase[1] + '+' + possible_three_eutect_phase[2] + '+' + \
                           possible_three_eutect_phase[0]

            elif (possible_three_eutect_phase[2] + '+' + possible_three_eutect_phase[1] + '+' +
                  possible_three_eutect_phase[0]) in phase_area_kinds_combined:
                doc_name = possible_three_eutect_phase[2] + '+' + possible_three_eutect_phase[1] + '+' + \
                           possible_three_eutect_phase[0]

            elif (possible_three_eutect_phase[2] + '+' + possible_three_eutect_phase[0] + '+' +
                  possible_three_eutect_phase[1]) in phase_area_kinds_combined:
                doc_name = possible_three_eutect_phase[2] + '+' + possible_three_eutect_phase[0] + '+' + \
                           possible_three_eutect_phase[1]

            f = h5py.File(self.path + doc_name + '.h5', 'r')
            temp_area = f['features'][()]
            f.close()

            for i in range(temp_area.shape[0]):
                if temp_area[i, 0] != -1:
                    possible_ternary_phase_area[temp_area[i, 0]][temp_area[i, 1]][temp_area[i, 2]][
                        temp_area[i, 3]] = doc_name

            print(doc_name)

        self.possible_ternary_phases_area = possible_ternary_phase_area

        return possible_ternary_phase_area

    def find_eutectic_contents(self, t_tolerance=3):
        eutectic_reaction_list = []
        for i in self.liquid_boundary_list:
            temp = []
            if self.possible_eutectic_phases_area[i[0] - t_tolerance][i[1]][i[2]][i[3]]:
                temp.append([i[0] - t_tolerance, i[1], i[2], i[3]])
                temp.append(self.possible_eutectic_phases_area[i[0] - t_tolerance][i[1]][i[2]][i[3]])

            if temp:
                eutectic_reaction_list.append(temp)

        self.eutectic_contents = eutectic_reaction_list

        return eutectic_reaction_list

    def find_ternary_eutectic_contents(self, t_tolerance=5):
        ternary_reaction_list = []
        for i in self.liquid_boundary_list:
            temp = []
            if self.possible_ternary_phases_area[i[0] - t_tolerance][i[1]][i[2]][i[3]]:
                temp.append([i[0] - t_tolerance, i[1], i[2], i[3]])
                temp.append(self.possible_ternary_phases_area[i[0] - t_tolerance][i[1]][i[2]][i[3]])

            if temp:
                ternary_reaction_list.append(temp)

        self.ternary_eutectic_reaction_contents = ternary_reaction_list

        return ternary_reaction_list

    def t_extent_condition(self, extent=5):
        eutectic_content_fit_temper_con = []

        def if_enough_t_extent(content, possible_eutectic_phase_area):
            temp = content.copy()
            for t in range(extent):
                temp[0] = temp[0] - 1
                if not possible_eutectic_phase_area[temp[0]][temp[1]][temp[2]][temp[3]]:
                    return False
            return True

        for i in self.eutectic_contents:
            if if_enough_t_extent(i[0], self.possible_eutectic_phases_area):
                eutectic_content_fit_temper_con.append(i)

        self.eutectic_contents_fit_temper_con = eutectic_content_fit_temper_con

        return eutectic_content_fit_temper_con

    @staticmethod
    def draw_3D_fig(eutectic_contents):
        kinds = []
        for i in eutectic_contents:
            if not i[1] in kinds:
                kinds.append(i[1])

        fig = plt.figure(figsize=(6, 7))
        ax1 = plt.axes(projection='3d')
        color_index = 0
        #colors = ['#0000FF', '#00FFFF', '#000080']
        for i in kinds:
            x, y, z = [], [], []
            for j in eutectic_contents:
                if i == j[1]:
                    x.append(j[0][1])
                    y.append(j[0][2])
                    z.append(j[0][3])
            ax1.scatter3D(x, y, z, alpha=0.2)  #color=colors[color_index]
            color_index += 1
            break

        ax1.set_xlabel('Fe, at. %')
        ax1.set_ylabel('Ni, at. %')
        ax1.set_zlabel('Cr, at. %')
        ax1.view_init(elev=19, azim=-110)
        plt.legend(frameon=False)
        plt.show()
        #plt.savefig('D:\FeCoNiCrMn data\eutectic contents.png', dpi=300)


class eutectoidReaction:
    def __init__(self, path=None):
        self.eutectoid_reaction_lists = None
        self.possible_eutectoid_phases_area = None
        self.single_double_phases_list = None
        self.path = path

    def single_double_phases(self, phase_area_kinds):
        single_double_phases_list = []
        for i in phase_area_kinds:
            if not ('LIQUID' in i):
                if len(i) == 1 or len(i) == 2:
                    single_double_phases_list.append(i)

        self.single_double_phases_list = single_double_phases_list

        return single_double_phases_list

    def possible_eutectoid_phases_area_hash(self, phase_area_kinds_combined):
        possible_eutectoid_phase_area = [[[[0 for _ in range(101)] for _ in range(101)] for _ in range(101)] for _ in
                                         range(201)]
        for single_double_phase in self.single_double_phases_list:
            if len(single_double_phase) == 2:
                if (single_double_phase[0] + '+' + single_double_phase[1]) in phase_area_kinds_combined:
                    doc_name = single_double_phase[0] + '+' + single_double_phase[1]
                else:
                    doc_name = single_double_phase[1] + '+' + single_double_phase[0]

                f = h5py.File(self.path + doc_name + '.h5', 'r')
                temp_area = f['features'][()]
                f.close()

                for i in range(temp_area.shape[0]):
                    if temp_area[i, 0] != -1:
                        possible_eutectoid_phase_area[temp_area[i, 0]][temp_area[i, 1]][temp_area[i, 2]][
                            temp_area[i, 3]] = single_double_phase

                print(doc_name)

            else:
                f = h5py.File(self.path + single_double_phase[0] + '.h5', 'r')
                temp_area = f['features'][()]
                f.close()

                for i in range(temp_area.shape[0]):
                    if temp_area[i, 0] != -1:
                        possible_eutectoid_phase_area[temp_area[i, 0]][temp_area[i, 1]][temp_area[i, 2]][
                            temp_area[i, 3]] = single_double_phase

                print(single_double_phase[0])

        self.possible_eutectoid_phases_area = possible_eutectoid_phase_area

        return possible_eutectoid_phase_area

    def all_eutectoid_reaction(self):
        eutectoid_reaction_lists = []
        for Fe in range(101):
            print(Fe)
            for Ni in range(101):
                for Cr in range(101):
                    single_phase = []
                    double_phase = []
                    for T in range(201):
                        if self.possible_eutectoid_phases_area[T][Fe][Ni][Cr]:
                            if len(self.possible_eutectoid_phases_area[T][Fe][Ni][Cr]) == 2:
                                double_phase.append(self.possible_eutectoid_phases_area[T][Fe][Ni][Cr])
                            elif len(self.possible_eutectoid_phases_area[T][Fe][Ni][Cr]) == 1:
                                single_phase.append(self.possible_eutectoid_phases_area[T][Fe][Ni][Cr])
                    if single_phase and double_phase:
                        for single in single_phase:
                            for double in double_phase:
                                temp = []
                                if not (single[0] in double):
                                    temp.append(single)
                                    temp.append(double)
                                    eutectoid_reaction_lists.append(temp)

        self.eutectoid_reaction_lists = eutectoid_reaction_lists
        return eutectoid_reaction_lists


class precipitation:
    def __init__(self):
        self.possible_precipitation_kind = None

    def possible_precipitation(self, phase_area_kinds):
        possible_precipitation_kind = []
        for i in phase_area_kinds:
            if len(i) == 1:
                for j in phase_area_kinds:
                    if len(j) == 2 and (i[0] in j) and (not 'LIQUID' in j):
                        temp = []
                        temp.append(i)
                        temp.append(j)
                        possible_precipitation_kind.append(temp)

        self.possible_precipitation_kind = possible_precipitation_kind

        return possible_precipitation_kind

    def precipitation_area_hash(self, path, phase_area_kinds_combined):
        get_generated_vol = outputDataProcessor()
        get_generated_vol.import_phase_name_list("..\\generated phases.xlsx")
        phase_calc_model = calc_model()
        phase_calc_model.import_model("..\\model\\5.97e-5.h5")
        #phase_calc_model.import_encoder("D:\FeCoNiCrMn data\FeCrNiMn all data\model\with encoded\\1\encoder.h5")
        file_names = []
        for each_precipitation in self.possible_precipitation_kind:
            precipitation_hash = [[[[0 for _ in range(101)] for _ in range(101)] for _ in range(101)] for _ in
                                  range(101)]
            f = h5py.File(path + each_precipitation[0][0] + '.h5', 'r')
            single_phase_area = f['features'][()]
            f.close()
            print(each_precipitation[0][0])
            for i in phase_area_kinds_combined:
                if i.count('+') == 1:
                    if (each_precipitation[1][0] in i) and (each_precipitation[1][1] in i):
                        f = h5py.File(path + i + '.h5', 'r')
                        double_phase_area = f['features'][()]
                        f.close()
                        doc_name = i
                        print(doc_name)

            for i in range(single_phase_area.shape[0]):
                if single_phase_area[i, 0] != 111:
                    precipitation_hash[single_phase_area[i, 0]][single_phase_area[i, 1]][single_phase_area[i, 2]][single_phase_area[i, 3]] = each_precipitation[0][0]
            for i in range(double_phase_area.shape[0]):
                if double_phase_area[i, 0] != 111:
                    precipitation_hash[double_phase_area[i, 0]][double_phase_area[i, 1]][double_phase_area[i, 2]][double_phase_area[i, 3]] = doc_name

            contents_list = []
            for Fe in range(101):
                print(Fe)
                for Ni in range(101):
                    for Cr in range(101):
                        flag = 0
                        temp = []
                        for T in range(101):
                            if precipitation_hash[T][Fe][Ni][Cr]:
                                if '+' in precipitation_hash[T][Fe][Ni][Cr]:
                                    flag = 1
                                    temp.append([T, Fe, Ni, Cr])
                                if (flag == 1 or flag == 2) and (not '+' in precipitation_hash[T][Fe][Ni][Cr]):
                                    if flag == 1:
                                        for i in temp:
                                            contents_list.append(i)
                                        flag = 2
                                    contents_list.append([T, Fe, Ni, Cr])

            if contents_list:
                contents_np = np.array(contents_list)
                contents_np_temp = np.array(contents_list)
                contents_np = contents_np / 100
                #contents_np = np.insert(contents_np, 2, 0, axis=1)
                contents_np = np.insert(contents_np, 5, 0, axis=1)

                for i in range(contents_np.shape[0]):
                    contents_np[i, 4] = 1 - sum(contents_np[i, 1:4])

                contents_np = abs(contents_np)
                output_y = phase_calc_model.model.predict(contents_np)
                output_y = output_normalize(output_y)
                phase_with_vol = get_generated_vol.get_generated_phases_with_vol(output_y, 0)

                if each_precipitation[1][0] == each_precipitation[0][0]:
                    precipitation_kind_name = each_precipitation[1][1]
                elif each_precipitation[1][1] == each_precipitation[0][0]:
                    precipitation_kind_name = each_precipitation[1][0]

                precipitation_vol_list = []
                for i in range(contents_np.shape[0]):
                    if precipitation_kind_name in phase_with_vol[i]:
                        precipitation_vol_list.append(phase_with_vol[i][phase_with_vol[i].index(precipitation_kind_name)-1])
                    else:
                        precipitation_vol_list.append(0)

                precipitation_vol_np = np.array(precipitation_vol_list)
                precipitation_vol_np = np.expand_dims(precipitation_vol_np, axis=1)
                contents_np_temp = np.delete(contents_np_temp, 0, axis=1)
                precipitation_vol_np = np.append(contents_np_temp, precipitation_vol_np, axis=1)
                zero_index_list = []
                for i in range(precipitation_vol_np.shape[0]):
                    if precipitation_vol_np[i, -1] == 0:
                        zero_index_list.append(i)

                precipitation_vol_np = np.delete(precipitation_vol_np, zero_index_list, axis=0)

                f = h5py.File('D:\FeCoNiCrMn data\FeCrNiMn all data\high dim phase diagram\precipitation phase with volume\\' + each_precipitation[0][0] + '-' + doc_name + '.h5', 'w')
                f.create_dataset('features', data=precipitation_vol_np)
                f.close()
                file_names.append(each_precipitation[0][0] + '-' + doc_name)

        return file_names
# a = eutecticReaction('D:\FeCoNiCrMn data\FeCrNiMn all data\high dim phase diagram\each phase kind data\each kind with divided area\\')
# b = a.liquid_lower_boundary()
