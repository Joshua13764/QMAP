import matplotlib
import matplotlib.pyplot as plt


class PlotSettings:
    @staticmethod
    def load_default() -> None:
        # Plot settings
        matplotlib.use("Agg")
        plt.style.use('science')
        plt.rcParams["figure.figsize"] = (7, 7 * ((5**0.5 - 1) / 2))
        DPI = 400
        plt.rcParams["figure.dpi"] = 400
        plt.ioff()
