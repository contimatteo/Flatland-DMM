import json

import numpy as np
from matplotlib import pyplot as plt


def moving_average(window_size):
    return lambda d: (d[0], np.convolve(d[1], np.ones(window_size) / window_size, mode='same'))


FILE_PATH = 'plot_data.json'
JSON_KEY = 'example_key'
PLOT_PARAMS = dict(
    x_label='N Episodes',
    y_label='% Agent completion',
    fig_title='',
    save_as=None,
    # x_axis=np.arange(0, 41, 5),
    y_axis=np.arange(0, 101, 10),
    processing_function=moving_average(100)
)


# -----------------------------------------------------------------


class Plotter:
    def __init__(self, data: np.ndarray, x_label: str, y_label: str, fig_title, save_as, x_axis=None, y_axis=None,
                 processing_function=lambda t: t):
        self.x_label, self.y_label = x_label, y_label
        self.x_axis, self.y_axis = x_axis, y_axis
        self.fig_title = fig_title
        self.processing_function = processing_function
        self.save_as = save_as
        self.data = data

    def plot(self):
        # Data
        x, y = self.data.T
        x, y = self.processing_function((x, y))

        # Plot
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel=self.x_label, ylabel=self.y_label, title=self.fig_title)
        if self.x_axis is not None:
            ax.set_xticks(self.x_axis)
        if self.y_axis is not None:
            ax.set_yticks(self.y_axis)

        # Save & show
        if self.save_as:
            fig.savefig(self.save_as)
        plt.show()

    @staticmethod
    def from_json(file_path, json_key, **kwargs):
        with open(file_path) as file:
            # Load json from file
            j = json.loads(file.read())
            data = j[json_key]
            assert data is not None
            # Convert to np and return
            data = np.array(data)
            assert len(data.shape) == 2
            print(f'Loaded data with shape: {data.shape}')
            return Plotter(data, **kwargs)

    @staticmethod
    def from_dummy_data(**kwargs):
        data = np.array([
            np.arange(1000),
            np.random.uniform(0.0, 100.0, 1000)
        ]).T
        return Plotter(data, **kwargs)


# -----------------------------------------------------------------

if __name__ == '__main__':
    # plotter = Plotter.from_json(FILE_PATH, JSON_KEY, **PLOT_PARAMS)
    plotter = Plotter.from_dummy_data(**PLOT_PARAMS)
    plotter.plot()
