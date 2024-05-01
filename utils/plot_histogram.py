# generate code for retrieving pytorch tensor into numpy array and matplot them in histogram
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# make the x-axis of the histogram follows logarithmic scale
# https://matplotlib.org/3.1.1/gallery/statistics/hist.html#sphx-glr-gallery-statistics-hist-py


def main():
    parser = argparse.ArgumentParser(
        description="Generate code for retrieving pytorch tensor into numpy array and matplot them in histogram"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="input torch file"
    )
    parser.add_argument(
        "-l", "--log", action="store_true", help="use logarithmic scale"
    )
    parser.add_argument("-t", "--title", type=str, help="title of the plot")
    parser.add_argument("-x", "--xlabel", type=str, help="label of x-axis")
    parser.add_argument("-y", "--ylabel", type=str, help="label of y-axis")
    parser.add_argument("-s", "--show", action="store_true", help="show the plot")
    args = parser.parse_args()

    name = "test_histo"
    # load data
    data = torch.load(args.input)

    # convert to numpy array
    data = data.numpy()

    # given above arguments, plot the input data with x-axis following logarithmic scale
    fig, ax = plt.subplots()
    num_bins = 50
    hist, bins, _ = plt.hist(data, bins=num_bins, density=True)
    # plt.hist(data)
    # plt.show()
    if args.log:
        ax.set_xscale("log")
        # bins_res = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        bins_res = np.logspace(bins[0], np.log10(bins[-1]), len(bins))
    else:
        bins_res = np.linspace(bins[0], bins[-1], len(bins))
    ax.hist(data, bins=bins_res)
    ax.set_title(args.title)  # title of the plot
    ax.set_xlabel(args.xlabel)  # label of x-axis
    ax.set_ylabel(args.ylabel)  # label of y-axis

    # save the plot
    fig.savefig("{}.png".format(name))

    # show the plot
    if args.show:
        plt.show()

    print("__plot generated")


if __name__ == "__main__":
    main()
