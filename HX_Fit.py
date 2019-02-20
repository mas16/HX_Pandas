"""
Script to read raw, uncleaned nmrPipe output (flat file) and extract
relevant data for fitting serially collected HX data. Number of data points
used in the fitting can be arbitrary.

Data are fit to standard three parameter exponential decay of form:

I(t) = I(0) * exp(-Rt) + b

where I is signal intensity, R is the decay rate, and b is the offset.

Fitting is done via non linear least squares optimization using scipy.

Generators are used to minimize the amount of data stored in memory.

by MAS 02/2019
"""

# Import libraries
import pandas as pd
import sys
import numpy as np
import pylab as plt
import scipy as sc
import scipy.optimize as so
import seaborn as sns

# Set parameters for graphics
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['ps.useafm'] = True
plt.rcParams['axes.linewidth'] = 3.0

# Set plotting style
sns.set_style("ticks")

###########################

# User Defined constants

# Path to time points
TIMES = "/Users/matthewstetz/Desktop/Parkin_HX_Grant/times.txt"

# Path to data directory
DIRECTORY = "/Users/matthewstetz/Desktop/Parkin_HX_Grant/ft/parkin_hx.tab"

# Guess for initial signal intensity (~1.0)
I0 = 1.0

# Guess for decay rate
RATE = 0.0005

# Guess for offset
OFFSET = 0.3

###########################

# Keep individual parameter initial guesses in tuple
PARAMS = (I0, RATE, OFFSET)


# Functions
def get_times(datapath):
    """Extract time points for external input file.
    Normally, the time points are extracted independently and
    stored as a text file.

    Arguments
    ---------
    :datapath: String, location of input file
    :return: list, time points (floats)
    """
    try:
        with open(datapath, "r") as f:
            times = [float(entry) for index, entry in enumerate(f)]
            return times
    except FileNotFoundError:
        print("ERROR: Input times file not found.")
        sys.exit()


def get_column_names(datapath, column_names=None):
    """Get the column names from the input flat file.

    The column names are designated by the entries of the row starting with
    "VARS." Since VARS is indexed, the column names are offset by 1.

    This is hard to deal with in Pandas so I prefer to just to use enumerate
    to find the VARS line. This also avoids storing the whole file in memory.

    Arguments
    -----------
    :datapath: string, location of input file.
    :column_names: default = None to avoid crash if VARS row not in file.
    :return: list, column names (strings)
    """
    with open(datapath, "r") as f:
        for index, line in enumerate(f):
            if line.startswith("VARS"):
                # Start index at 1 to correct offset
                column_names = line.split()[1:]
                break
        if not column_names:
            print("ERROR: No Column Nmes Found")
            sys.exit()
        return column_names


def get_df(df_path):
    """Get the data and store as pandas dataframe.
    Once the data are in a dataframe, the data are cleaned and normalized
    for fitting.

    Arguments
    ---------
    :df_path: string loction of input flat file.
    :return: dataframe of signal intensity
    """
    try:
        df = pd.read_table(df_path, sep="\s+", header=None, skiprows=12)
    except FileNotFoundError:
        print("ERROR: Input Data File Not Found.")
        sys.exit()

    df.columns = get_column_names(df_path)
    intensity = df.loc[:, df.columns.str.startswith("Z_")]
    intensity.loc[:, "Z_A0"] = intensity.loc[:, "Z_A0"]*2
    intensity = intensity.apply(lambda x: x/2.0)
    return intensity


def threeparam_fit(x, p):
    """Three parameter single exponential model.

    Parameters
    ----------
    :p[0]: float, initial signal intensity, I(0)
    :p[1]: float, decay rate
    :p[2]: float, offset term

    Arguments
    ---------
    :x: array of time points (float)
    :p: array of parameters (float)
    :return: model
    """
    return (p[0] * sc.exp(sc.array(x) * -p[1])) + p[2]


def threeresiduals(p, y, x):
    """Loss function to minimize for non-linear least squares optimization.

    Arguments
    ---------
    :p: array of parameters (float)
    :y: array of observed data points to model
    :x: array of time points (float)
    :return: loss function
    """
    return y - threeparam_fit(x, p)


def fit_plot(int_df, params, smooth_time, data_times):
    """Fit the data and plot.
    Use iterrows generator to minimize memory load.

    Arguments
    ---------
    :int_df: dataframe of signal intensities
    :params: array of guesses for fitting (float)
    :smooth_time: array of time points for model (float)
    :data_times: array of time points (float)
    :return: None
    """
    for index, row in int_df.iterrows():
        lsq = so.leastsq(threeresiduals, params,
                         args=(row.tolist(), data_times), full_output=True)
        smooth_fit = threeparam_fit(smooth_time, lsq[0])
        plt.plot(smooth_time, smooth_fit, "-r", linewidth=3.0)
        plt.plot(data_times, np.array(row.tolist()), "ok", markersize=10)
        plt.title(str(index))
        plt.axis([0, 1.2*max(data_times), 0.0, 1.2])
        plt.ylabel("Normalized Intensity (a.u.)", fontsize=20,
                   fontweight="bold")
        plt.xlabel("Time (s)", fontsize=20, fontweight="bold")
        plt.tick_params(width=3, length=5, labelsize=16)
        sns.despine()
        plt.show()


def run(times=TIMES, directory=DIRECTORY, params=PARAMS):
    """This function will execute the major functions.
    1. Get the time points
    2. Get the raw data
    3. Genereate array for visualizing the modeled function
    4. Fitting and plotting data

    Arguments
    ---------
    :times: array of time points(float)
    :directory: string, location of data file
    :params: array of initial guesses for least squares (float)
    :return: None
    """
    data_times = get_times(times)
    intensity = get_df(directory)
    smooth_time = np.arange(0, 1.2 * max(data_times), 1000)
    fit_plot(intensity, params, smooth_time, data_times)


if __name__ == '__main__':
    run()
