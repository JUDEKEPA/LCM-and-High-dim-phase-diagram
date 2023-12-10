import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
from NN_utility.customclass import WeightedSumOfExperts, NormalizeActivation, add_normalize_activation


class phaseDiagram:
    def __init__(self, elements=None, x_names=None, generated_phase_path=None):
        self.elements = elements
        self.min_T = None
        self.max_T = None
        self.t_increment = None
        self.x_names = x_names
        self.x_element = None
        self.x_element_min = None
        self.x_element_max = None
        self.x_increment = None
        self.fixed_elements = {}
        self.generated_phase_path = generated_phase_path
        self.diagram_metrics = None
        self.predict_phases = None
        self.phase_names_metrics = None

    def temperate(self, min_T, max_T, increment=3):
        self.min_T = min_T
        self.max_T = max_T
        self.t_increment = increment

    def fixed_elements_vals(self, element, val):
        self.fixed_elements[element] = val

    def x_axis_element(self, element, min, max, increment=0.005):
        self.x_element = element
        self.x_element_min = min
        self.x_element_max = max
        self.x_increment = increment

    def phase_diagram_metrics(self):
        t_axis = np.arange(self.min_T, self.max_T, self.t_increment)
        x_axis = np.arange(self.x_element_min, self.x_element_max, self.x_increment)
        diagram_metrics = np.zeros(shape=(t_axis.shape[0] * x_axis.shape[0], len(self.x_names)))
        added_elements = []  # Insert fixed elements
        for i in self.fixed_elements:
            index = self.x_names.index(i)
            added_elements.append(i)
            diagram_metrics[:, index] = self.fixed_elements[i]

        #  Fill diagram_metrics

        x_element_index = self.x_names.index(self.x_element)
        i, j = 0, 0
        while i < t_axis.shape[0] * x_axis.shape[0]:
            for k in range(x_axis.shape[0]):
                diagram_metrics[i, 0] = t_axis[j]
                diagram_metrics[i, x_element_index] = x_axis[k]
                i += 1
            j += 1

        added_elements.append(self.x_element)  # Insert element on x-axis
        exception_element = self.elements.copy()
        for i in added_elements:  # Find the left element
            exception_element.pop(exception_element.index(i))

        exception_element_index = self.x_names.index(str(exception_element[0]))
        for i in range(t_axis.shape[0] * x_axis.shape[0]):
            diagram_metrics[i, exception_element_index] = 1 - sum(diagram_metrics[i, 1:])

        self.diagram_metrics = diagram_metrics

    def phase_calculation(self, model, encoder=None, encoded=False):
        input_diagram_metrics = self.diagram_metrics.copy()
        input_diagram_metrics[:, 0] = (input_diagram_metrics[:, 0] - 673.15) / 2000
        if encoded:
            input_data = encoder.predict(input_diagram_metrics)
        else:
            input_data = input_diagram_metrics

        predict_phases = model.predict(input_data)
        for i in range(predict_phases.shape[0]):
            predict_phases[i] = predict_phases[i] / sum(predict_phases[i])
        self.predict_phases = predict_phases

    def predict_phases_names(self, tolerance=0.02):
        phase_names = list(pd.read_excel(self.generated_phase_path).columns)  # Import generated phase
        diagram_names = []
        for i in range(self.predict_phases.shape[0]):
            temp = []
            for j in range(len(phase_names)):
                if self.predict_phases[i, j] > tolerance:
                    temp.append(phase_names[j])
            diagram_names.append(temp)

        combined_diagram_names = []
        for i in range(self.predict_phases.shape[0]):
            temp = ''
            for j in diagram_names[i]:
                temp += j
                temp += '+'
            temp = temp[:-1]
            combined_diagram_names.append(temp)


        self.phase_names_metrics = combined_diagram_names

    def draw_phase_diagram(self):
        x_element_index = self.x_names.index(self.x_element)
        x = self.diagram_metrics[:, x_element_index]
        y = self.diagram_metrics[:, 0]
        kinds = set(self.phase_names_metrics)
        colors = {}
        color_names = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf',  # blue-teal
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
            '#c49c94',  # light brown
            '#f7b6d2',  # light pink
            '#c7c7c7',  # light gray
            '#dbdb8d',  # light yellow-green
            '#e69f00',  # orange
            '#56b4e9',  # sky blue
            '#009e73',  # bluish green
            '#f0e442',  # yellow
            '#0072b2',  # blue
            '#d55e00',  # vermillion
            '#cc79a7',  # reddish purple
            '#6A3D9A',  # plum
            '#B15928',  # rusty orange
            '#33a02c',  # forest green
            '#FF0000',  # Red
            '#00FF00',  # Lime
            '#0000FF',  # Blue
            '#FFFF00',  # Yellow
            '#00FFFF',  # Cyan / Aqua
            '#FF00FF',  # Magenta / Fuchsia
            '#800000',  # Maroon
            '#808000',  # Olive
            '#008000',  # Green
            '#800080',  # Purple
            '#008080',  # Teal
            '#000080',  # Navy
            '#FA8072',  # Salmon
            '#FFA500',  # Orange
            '#20B2AA',  # Light Sea Green
            '#778899',  # Light Slate Gray
            '#B0C4DE',  # Light Steel Blue
            '#5F9EA0',  # Cadet Blue
            '#4682B4',  # Steel Blue
            '#9ACD32',  # Yellow Green
            '#32CD32',  # Lime Green
            '#FFD700',  # Gold
            '#DA70D6',  # Orchid
            '#8B0000',  # Dark Red
            '#556B2F',  # Dark Olive Green
            '#FF69B4',  # Hot Pink
            '#FFDAB9',  # Peach Puff
            '#CD853F'  # Peru
        ]
        color_index = 0
        for i in kinds:
            colors[i] = color_names[color_index]
            color_index += 1

        plt.figure(figsize=(5, 4))
        phase_color = [colors[phase] for phase in self.phase_names_metrics]
        scatter = plt.scatter(x*100, y, c=phase_color, marker='o')
        plt.tick_params(labelsize=15)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=phase, markersize=10, markerfacecolor=colors[phase]) for phase
            in colors]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0), loc=3)
        plt.subplots_adjust(right=0.5)
        plt.show()

    def integrate_draw(self, model, encoder=None, encoded=False):
        self.phase_diagram_metrics()
        self.phase_calculation(model, encoder, encoded)
        self.predict_phases_names()
        self.draw_phase_diagram()


FeNiCrMn = phaseDiagram(['Fe', 'Ni', 'Cr', 'Mn'], ['T', 'Fe', 'Ni', 'Cr', 'Mn'], "D:\FeCoNiCrMn data\FeCrNiMn all data\generated phases.xlsx")
FeNiCrMn.temperate(700, 2400, 2)
FeNiCrMn.fixed_elements_vals('Fe', 0.2)
FeNiCrMn.fixed_elements_vals('Mn', 0.2)
FeNiCrMn.x_axis_element('Ni', 0, 0.6, 0.001)


model = load_model("D:\FeCoNiCrMn data\FeCrNiMn all data\model\\5.97e-5.h5")