import numpy as np
import pandas as pd
from keras.models import load_model


def accuracy_validation(real, predict, tolerance=0.01):
    data_num = real.shape[0]
    phase_num = real.shape[1]

    accurate_count = 0

    for i in range(data_num):
        temp_bool = True
        for j in range(phase_num):
            if real[i, j] >= tolerance and predict[i, j] < tolerance:
                temp_bool = False
                break
            elif real[i, j] < tolerance and predict[i, j] >= tolerance:
                temp_bool = False
                break
        if temp_bool:
            accurate_count += 1

    return accurate_count / data_num * 100


class outputDataProcessor:
    def __init__(self):
        self.phase_name_list = None

    def import_phase_name_list(self, path):
        df = pd.read_excel(path)
        self.phase_name_list = list(df.columns)

    def get_generated_phases(self, phase_vol_row, tolerance=0.03):
        generated_phase_list = []
        for i in range(phase_vol_row.shape[0]):
            temp = []
            for j in range(phase_vol_row.shape[1]):
                if phase_vol_row[i, j] > tolerance:
                    temp.append(self.phase_name_list[j])
            generated_phase_list.append(temp)

            if i % 1000000 == 0:
                print(i)

        return generated_phase_list

    def get_generated_phases_with_vol(self, phase_vol_row, tolerance=0.03):
        generated_phase_vol_list = []
        for i in range(phase_vol_row.shape[0]):
            temp = []
            for j in range(phase_vol_row.shape[1]):
                if phase_vol_row[i, j] > tolerance:
                    temp.append(phase_vol_row[i, j])
                    temp.append(self.phase_name_list[j])
            generated_phase_vol_list.append(temp)

            if i % 1000000 == 0:
                print(i)

        return generated_phase_vol_list
