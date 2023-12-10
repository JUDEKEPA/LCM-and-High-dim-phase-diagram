import numpy as np
from basecode.calcpha import calc_model
from basecode.generatephase import outputDataProcessor
import h5py


class hiPhaData(calc_model, outputDataProcessor):
    def __init__(self):
        super().__init__()
        self.t_increment = None
        self.x_increment = None
        self.all_outputs = None
        self.all_inputs = None
        self.inputs_int = None
        self.all_output_generated_phases = None
        self.phase_kinds = None

    def ergodic_array(self, x_increment=0.01, t_increment=0.005, calc=True):
        self.x_increment = x_increment
        self.t_increment = t_increment
        ergo_array = []
        start_point = [0, 0, 0, 0]
        while start_point[1] <= 1.00005:
            start_point[2] = 0
            while start_point[2] <= 1.00005:
                start_point[3] = 0
                while start_point[3] <= 1.00005:
                    start_point[0] = 0
                    while start_point[0] <= 1.00005:
                        if self.is_valid_point(start_point):
                            ergo_array.append(start_point.copy())
                        start_point[0] += t_increment
                    start_point[3] += x_increment
                start_point[2] += x_increment
            start_point[1] += x_increment
            print(start_point[1])

        ergo_array = np.array(ergo_array)
        if calc:
            left_one_concentration = []
            for i in range(ergo_array.shape[0]):
                left_one_concentration.append(1 - sum(ergo_array[i, 1:]))

            left_one_concentration = np.array(left_one_concentration)
            left_one_concentration = np.expand_dims(left_one_concentration, axis=1)
            ergo_array = np.append(ergo_array, left_one_concentration, axis=1)

        self.all_inputs = ergo_array

    def output_calculate(self, batch_size):
        outputs = self.model.predict(self.all_inputs[:1, :])
        for i in range(0, self.all_inputs.shape[0], batch_size):
            temp = self.model.predict(self.all_inputs[i:i + batch_size, :])
            outputs = np.append(outputs, temp, axis=0)
            print(i)
        outputs = np.delete(outputs, 0, axis=0)

        self.all_outputs = outputs

    def get_output_generated_phases(self, tolerance=0.03):
        all_output_generated_phases_list = self.get_generated_phases(self.all_outputs, tolerance)
        all_output_generated_phases = []
        for phase_status in all_output_generated_phases_list:
            temp = ''
            for phase in phase_status:
                temp += phase
                temp += '+'
            temp = temp[:-1]
            all_output_generated_phases.append(temp)

        all_output_generated_phases = np.array(all_output_generated_phases)
        all_output_generated_phases = np.expand_dims(all_output_generated_phases, axis=1)
        self.all_output_generated_phases = all_output_generated_phases

    def get_phase_kinds(self):
        phase_kinds = []
        for i in range(self.all_output_generated_phases.shape[0]):
            if not self.all_output_generated_phases[i, 0] in phase_kinds:
                phase_kinds.append(self.all_output_generated_phases[i, 0])

        phase_kinds = np.array(phase_kinds)
        phase_kinds = np.expand_dims(phase_kinds, axis=1)
        self.phase_kinds = phase_kinds

    def make_inputs_into_int(self):
        inputs_int = self.all_inputs
        inputs_int = np.delete(inputs_int, inputs_int.shape[1] - 1, axis=1)
        inputs_int[:, 0] = inputs_int[:, 0] * (1 / self.t_increment) + 0.4
        inputs_int[:, 1:] = inputs_int[:, 1:] * (1 / self.x_increment) + 0.4
        inputs_int = inputs_int.astype(int)
        self.inputs_int = inputs_int

    def save_inputs_array(self, path):
        f = h5py.File(path + '\\inputs array.h5', 'w')
        f.create_dataset('features', data=self.all_inputs)
        f.close()

    def save_phase_vol_data(self, path):
        f = h5py.File(path + '\\phase vol data.h5', 'w')
        f.create_dataset('features', data=self.all_outputs)
        f.close()

    def save_generated_phases(self, path):
        np.save(path + '\\phase name data.npy', self.all_output_generated_phases)

    def save_int_inputs(self, path):
        f = h5py.File(path + '\\int inputs for hash.h5', 'w')
        f.create_dataset('features', data=self.inputs_int)
        f.close()

    def save_phase_kinds(self, path):
        np.save(path + '\\phase kinds.npy', self.phase_kinds)

    def integrate_save(self, path):
        if self.all_inputs:
            self.save_inputs_array(path)
            self.make_inputs_into_int()
        if self.inputs_int:
            self.save_int_inputs(path)
        if self.all_outputs:
            self.save_phase_vol_data(path)
        if self.all_output_generated_phases:
            self.save_generated_phases(path)
        if self.phase_kinds:
            self.save_phase_kinds(path)

    def discrete_phase_status(self, path):
        for i in range(self.phase_kinds.shape[0]):
            phase_area = self.phase_kinds[i, 0]
            index = []
            for j in range(self.all_output_generated_phases.shape[0]):
                if self.all_output_generated_phases[j, 0] == phase_area:
                    index.append(j)
            temp = self.inputs_int[index, :]
            f = h5py.File(path + str(
                phase_area) + '.h5', 'w')
            f.create_dataset('features', data=temp)
            f.close()
            print(i)

    @staticmethod
    def is_valid_point(point):
        """Check if point is valid"""
        #  Normalized temperature should between 0 to 1
        if point[0] <= 1.0005 and point[0] >= 0:
            if sum(point[1:]) <= 1.0005:
                return True
            else:
                return False
        else:
            return False



