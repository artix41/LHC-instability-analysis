import pytimber
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy import signal
import sys
import argparse
import datetime
import time

from pylab import rcParams

NB_TURNS_PER_SECOND = 11800

def get_growth_rate(t1, t2, plane="H", beam=1):
    """ Get the growth rate of an instability occuring from t1 to t2"""

    plane_number = 1 if plane=='H' else 2 # useful for the name of our variable
    variable = 'LHC.BQBBQ.CONTINUOUS_HS.B' + str(beam) + ':EIGEN_AMPL_' + str(plane_number)

    print("Getting the data...")
    vn=[variable]
    data=db.get(vn,t1,t2)
    if (len(data[variable][0]) == 0): # if no data (empty array)
        print("No data available for BBQ amplitude in that time period (%s)" % str(variable))
        print("\033[91m" + "[-] Fail")
        exit()
    x_v = (data[variable][0]-data[variable][0][0])
    y_v = data[variable][1]
    print("[+] Success")

    # We now try to get the right limit of the exponential (on the x-axis)
    # For that, we look for the maximum of the derivative

    # We first need to apply a filter to remove noise
    print("Applying a gaussian filter...")
    size_window = len(y_v)//10
    std = size_window//3
    window = signal.gaussian(size_window, std=std)
    y_conv = np.convolve(y_v, window, 'valid') # gaussian filter
    print("[+] Success")

    # We then apply a second filter to the derivative, and get its argmax
    print("Getting the right boundary...")
    y_prime_v = np.diff(y_conv, n=1)
    y_prime_conv = np.convolve(y_prime_v, window, 'valid')
    right_bound = np.argmax(y_prime_conv) + size_window # We add size_window to compensate the two convolutions
    print("[+] Success")

    print("Fitting the curve...")
    # Initialization parameters : a naive exponential who pass by the two limit points
    x1, x2 = x_v[0], x_v[right_bound]
    y1, y2 = y_v[0], y_v[right_bound]
    d = (x1 - x2) / (np.log(y1) - np.log(y2)) # can be obtained by just writing down the equation system
    x0 = x1 - d * np.log(y1)
    def expo(x, d, c, x0):
        return np.exp((x-x0)/d) + c
    popt, pcov = curve_fit(expo, x_v[:right_bound], y_v[:right_bound], maxfev=10000, p0=[d, y_v[0], x0])
    print("[+] Success")

    print("Found growth rate : ", popt[0])
    plt.plot(x_v, y_v, label="Real data")
    plt.plot(x_v[:right_bound], expo(x_v[:right_bound],*popt), c="red", label="Exponential curve fitted")

    return popt[0]

def plot_bbq_raw(t1, t2, plane="H", beam=1):
    variable = 'LHC.BQBBQ.CONTINUOUS_HS.B' + str(beam) + ':ACQ_DATA_' + plane
    vn = [variable]
    data = db.get(vn,t1,t2)

    if (len(data[variable][0]) == 0):
        print("No data available for BBQ raw data in that time period (%s)" % variable)
        print("\033[91m" + "[-] Fail")
        exit()

    y_v = data[variable][1].flatten()
    y_avg = data[variable][1].mean(axis=1)
    plt.plot(y_avg)

    return y_v

def plot_emittance(t1, t2, plane="H", beam=1):
    beam_letter = "L" if beam == 2 else "R"
    variable = 'LHC.BSRT.5%s4.B%d:FIT_SIGMA_%s'%(beam_letter, beam, plane)

    vn=[variable]
    data=db.get(vn,t1,t2)
    x_v = (data[variable][0]-data[variable][0][0])
    y_v = data[variable][1]
    print(x_v.shape)
    print(y_v.shape)

    plt.plot(x_v, y_v)

    return y_v

if __name__ == "__main__":
    db=pytimber.LoggingDB()
    parser = argparse.ArgumentParser(description='Display interesting plots for instability analysis between two times.',
                                     epilog='Exemple : python3 instability-analysis.py "2017-06-30 16:26:00" "2017-06-30 16:32:00" -b 1 -p H',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('t1', metavar='t1', nargs='?', default=datetime.datetime.now() - datetime.timedelta(minutes=2),
                        help='Date-time string representing the first time to give to PyTimber')
    parser.add_argument('t2', metavar='t2', nargs='?', default=datetime.datetime.now(),
                        help='Date-time string representing the second time to give to PyTimber')
    parser.add_argument('-p','--plane', metavar='Plane', nargs='?', default='H',
                        help='H for horizontal, V for vertical')
    parser.add_argument('-b','--beam', metavar='Beam', type=int, nargs='?', default=1,
                        help='1 for beam 1, 2 for beam 2')

    args = parser.parse_args()

    # plt.ion()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # plane_number = 1 if args.plane=='H' else 2 # useful for the name of our variable
    # variable = 'LHC.BQBBQ.CONTINUOUS_HS.B' + str(args.beam) + ':EIGEN_AMPL_' + str(plane_number)
    # vn = [variable]
    #
    # data = db.get(vn, datetime.datetime.now() - datetime.timedelta(minutes=2, seconds=1), datetime.datetime.now() - datetime.timedelta(minutes=1, seconds=1))
    # y_v = data[variable][1]
    # line1, = ax.plot(y_v)
    #
    # while(True):
    #     data = db.get(vn, datetime.datetime.now() - datetime.timedelta(minutes=2, seconds=1), datetime.datetime.now() - datetime.timedelta(minutes=1, seconds=1))
    #     y_v = data[variable][1]
    #     line1.set_ydata(y_v)
    #     fig.canvas.draw()

    plt.suptitle("Beam " + str(args.beam) + ", Plane " + args.plane)


    # BBQ Raw Data
    plt.subplot(2,2,1)
    y_raw = plot_bbq_raw(args.t1, args.t2, args.plane, args.beam)
    #y_raw = plot_bbq_raw(datetime.datetime.now() - datetime.timedelta(minutes=3), datetime.datetime.now() - datetime.timedelta(minutes=1), args.plane, args.beam)
    plt.title("BBQ Raw Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")

    # Fourier transform (spectrogram)
    plt.subplot(2,2,2)
    plt.specgram(y_raw, Fs=1, NFFT=2048)
    plt.title("Spectrogram")

    # BBQ Amplitude and Growth rate
    plt.subplot(2,2,3)
    growth_rate = get_growth_rate(args.t1, args.t2, args.plane, args.beam)
    plt.title("BBQ Eigen Amplitude, growth rate : %.3f" % growth_rate)
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

    'LHC.BCTFR.A6R4.B%d:BEAM_INTENSITY'%beam
    'LHC.BSRT.5%s4.B%d:FIT_SIGMA_H'%(beam_device_list[beam-1],beam)
