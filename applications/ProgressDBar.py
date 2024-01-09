class color:
    PURPLE = "\033[1;35;48m"
    CYAN = "\033[1;36;48m"
    BOLD = "\033[1;37;48m"
    BLUE = "\033[1;34;48m"
    GREEN = "\033[1;32;48m"
    YELLOW = "\033[1;33;48m"
    RED = "\033[1;31;48m"
    BLACK = "\033[1;30;48m"
    UNDERLINE = "\033[4;37;48m"
    END = "\033[1;37;0m"


import IPython
import time
import numpy as np
import matplotlib.pyplot as plt


class ProgressTrain:
    def __init__(self, imax, Size=50, num_update=100, names=[]):
        self.imax = imax
        self.i = 0
        self.Scale = Size / imax
        self.Size = Size
        self.value_hist = []
        self.names = names
        self.update = int(imax // num_update + 1)

        # plt.ion()
        # self.fig,  self.axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        self.out = display(IPython.display.Pretty("Starting"), display_id=True)  #
        time.sleep(2)

        self.colorList = [
            color.RED,
            color.YELLOW,
            color.PURPLE,
            color.BLUE,
            color.CYAN,
            color.GREEN,
        ]
        self.colorList = [x.replace("8m", "8m") for x in self.colorList]

        print("||    ||          Loss          ||        Accuracy        ||")
        print("||    ||   Train   ||   Test    ||   Train   ||   Test    ||")
        time.sleep(1)

        self.Update()

    def Update(self, x=[]):
        if self.i % self.update == 0:
            Color = self.colorList[
                min(
                    len(self.colorList) - 1,
                    round(self.i / self.imax * (len(self.colorList) - 1)),
                )
            ]

            string = (
                Color
                + "["
                + "%" * round(self.i * self.Scale)
                + "-" * round((self.imax - self.i) * self.Scale)
                + "] "
                + str(round(100 * self.i / (self.imax), 2))
                + "% "
                + color.END
            )
            string += " " + color.BOLD
            for value in x:
                string += " || " + self.Show(value)

            string += color.END + "  "

            self.out.update(IPython.display.Pretty(string))
            time.sleep(0.1)

        self.i += 1

    def Show(self, x, n=9):
        if n < 5:
            assert "To short to show"
        if type(x) == int or type(x) == float:
            if x >= (10**5) or x <= (10 ** (-5)):
                x = ("{:." + str(n - 4) + "e}").format(x)
            else:
                x = str(x)[:n]
        else:
            x = str(x)[:n]
        return x + " " * (len(x) - n)

    def End(self, x=[], plot=False):
        self.i -= 1

        if self.i == self.imax:
            string = color.GREEN + "||DON" + color.END + "" + color.BOLD

        else:
            string = color.YELLOW + "||DON" + color.END + "" + color.BOLD
        for value in x:
            string += " || " + self.Show(value)
        self.i = 0
        self.Update()

        print(string)
        self.value_hist.append(x)

    def get(self, input_names):
        data = np.transpose(np.array(self.value_hist))
        if not isinstance(input_names, list):
            input_names = [input_names]
        # Find indices of the input names
        indices = [self.names.index(name) for name in input_names if name in self.names]
        # Extract and return the data for these indices
        return [data[index] for index in indices]

    def plot_data_by_names(self, input_names):
        data = np.transpose(np.array(self.value_hist))
        # Ensure input_names is a list
        if isinstance(input_names, str):
            input_names = [input_names]

        # Plot data for each name in input_names
        for name in input_names:
            if name in self.names:
                index = self.names.index(name)
                plt.plot(data[index], label=name)

        # Set plot features
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Data Plot")
        plt.legend()
        plt.grid()
        plt.show()
