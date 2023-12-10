import numpy as np
from basecode.generatephase import outputDataProcessor
import sys
import h5py
from NN_utility.customclass import WeightedSumOfExperts, NormalizeActivation
from keras.models import load_model
from basecode.highdimsphadia import hiPhaData
from basecode.calcpha import calc_model


FeCrNiMn = hiPhaData()

sys.setrecursionlimit(100000)
model1 = calc_model()
model1.import_model("..\\model\\5.97e-5.h5")
FeCrNiMn.ergodic_array(0.01, calc=True)
FeCrNiMn.model = model1.model
FeCrNiMn.output_calculate(100000)

FeCrNiMn.import_phase_name_list("..\\generated phase.xlsx")
FeCrNiMn.get_output_generated_phases()
FeCrNiMn.integrate_save(save_path)
FeCrNiMn.discrete_phase_status(save_path)
