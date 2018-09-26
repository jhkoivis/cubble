import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
import os
import json

##########
# Fitting the data -- Least Squares Method
##########

# Power-law fitting is best done by first converting
# to a linear equation and then fitting to a straight line.
# Note that the `logyerr` term here is ignoring a constant prefactor.
#
#  y = a * x^b
#  log(y) = log(a) + b*log(x)
#

def fit_to_data(xdata, ydata):
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    yerr = np.ones(ydata.shape)
    logyerr = yerr / ydata
    
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    powerlaw = lambda x, amp, index: amp * (x**index)
    
    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)
    
    pfinal = out[0]
    covar = out[1]
    print pfinal
    print covar
    
    index = pfinal[1]
    amp = 10.0**pfinal[0]
    
    indexErr = np.sqrt( covar[1][1] )
    ampErr = np.sqrt( covar[0][0] ) * amp
    
    ##########
    # Plotting data
    ##########
    
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(xdata, powerlaw(xdata, amp, index))     # Fit
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    plt.title('Best Fit Power Law')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(1, 11)

    plt.subplot(2, 1, 2)
    plt.loglog(xdata, powerlaw(xdata, amp, index))
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    plt.xlim(1.0, 11)

    plt.show()

def plot_data_loglog(data_file, json_file, ax):
    print("Plotting data from \"" + str(data_file) + "\" using \"" + str(json_file) + "\" for some parameters.")
    data = np.loadtxt(data_file)
    x = data[:, 0]
    y = data[:, 1]
            
    with open(json_file, 'r') as f:
        decoded_json = json.load(f)
            
    phi = decoded_json["PhiTarget"]
    kappa = decoded_json["Kappa"]
    label_str = r"$\phi=$" + str(phi) + r", $\kappa=$" + str(kappa)
    
    ax.loglog(x, y, '-', linewidth=1.5, label=label_str)

def plot_relative_radius(ax, parent_dir, data_file_name, json_file_name, num_plots):
    
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.set_xlim(1, 2300)
    ax.set_ylim(0.65, 10)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\frac{R}{<R_{in}>}$")
    #ax.set_axis_bgcolor((160 / 255.0, 198 / 255.0, 255 / 255.0))
    ax.grid(1)

    child_dirs = [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    json_files = []
    data_files = []
    phis = []

    # Gather files and sort them according to phi
    for dir in child_dirs:
        for root, dirs, files in os.walk(os.path.join(parent_dir, dir)):
            if not files:
                continue

            if data_file_name in files and json_file_name in files:
                data_file = os.path.join(root, data_file_name)
                json_file = os.path.join(root, json_file_name)
        
                if not os.path.exists(data_file) or not os.path.exists(json_file):
                    print("Path constructed from given arguments does not exist. Path: \"" + str(root) + "\".")
                    print("Data file path: \"" + str(data_file) + "\".")
                    print("Json file path: \"" + str(json_file) + "\".")
                    print("Given arguments:\nparent directory: \"" + str(parent_dir) + "\"\ndata file name: \"" + str(data_file_name) + "\"\njson file name: \"" + str(json_file_name) + "\"\nnumber of files: " + str(num_files) + ".")
                    sys.exit(1)

                with open(json_file, 'r') as f:
                    decoded_json = json.load(f)
            
                phis.append(float(decoded_json["PhiTarget"]))
                json_files.append(json_file)
                data_files.append(data_file)

    indices = list(reversed(np.argsort(np.array(phis))))
    json_files = [json_files[idx] for idx in indices]
    data_files = [data_files[idx] for idx in indices]

    # Plot
    for i in range(min(len(data_files), max(num_plots, 0))):
        plot_data_loglog(data_files[i], json_files[i], ax)

    alpha1 = 0.62
    alpha2 = 0.48
    alpha3 = 0.5
    alpha4 = 0.39
    
    x1 = np.linspace(10, 600)
    y1 = pow(0.05 * x1, alpha1)
    
    x2 = np.linspace(10, 600)
    y2 = pow(0.0825 * x2, alpha2)
    
    x3 = np.linspace(10, 1800)
    y3 = pow(0.0138 * x3, alpha3)
    
    x4 = np.linspace(10, 1800)
    y4 = pow(0.0175 * x4, alpha4)

    line_color = (0, 0, 0)
    arrow_props_up=dict(arrowstyle='-', connectionstyle="angle,angleA=180,angleB=-90,rad=0")
    arrow_props_down=dict(arrowstyle='-', connectionstyle="angle,angleA=-90,angleB=180,rad=0")
    
    ax.loglog(x1, y1, '--', color=line_color, linewidth=2.0)
    ax.annotate(str(alpha1), xy=(x1[11], y1[15]))
    ax.annotate("", xy=(x1[13], y1[13]), xycoords='data', xytext=(x1[15], y1[15]), textcoords='data', arrowprops=arrow_props_up)
    
    ax.loglog(x2, y2, '--', color=line_color, linewidth=2.0)
    ax.annotate(str(alpha2), xy=(x2[2], y2[3]))
    ax.annotate("", xy=(x2[2], y2[2]), xycoords='data', xytext=(x2[3], y2[3]), textcoords='data', arrowprops=arrow_props_up)
    
    ax.loglog(x3, y3, '--', color=line_color, linewidth=2.0)
    ax.annotate(str(alpha3), xy=(x3[11], y3[15]))
    ax.annotate("", xy=(x3[13], y3[13]), xycoords='data', xytext=(x3[15], y3[15]), textcoords='data', arrowprops=arrow_props_up)
    
    ax.loglog(x4, y4, '--', color=line_color, linewidth=2.0)
    ax.annotate(str(alpha4), xy=(x4[2], y4[3]))
    ax.annotate("", xy=(x4[2], y4[2]), xycoords='data', xytext=(x4[3], y4[3]), textcoords='data', arrowprops=arrow_props_up)

    ax.legend()

    plt.show()
    
def main():
    arguments = sys.argv
    data_file_name = "collected_data.dat"
    json_file_name = "save.json"
    num_plots = 999
    
    if len(arguments) < 2:
        print("Too few arguments given. Give:\n\tthe parent folder of data files\n\t# of plots\n\tname of data file\n\tname of saved json file\nas arguments.")
        sys.exit(1)
    elif len(arguments) < 3:
        print("Using default names for data and json files, \"" + data_file_name + "\" and \"" + json_file_name + "\" respectively and plotting all data files.")
    elif len(arguments) < 4:
        num_plots = int(arguments[2])
        print("Using default names for data and json files, \"" + data_file_name + "\" and \"" + json_file_name + "\" respectively and plotting " + str(num_plots) + " data files.")
    elif len(arguments) < 5:
        num_plots = arguments[2]
        data_file_name = arguments[3]
        print("Using default name for json file: \"" + json_file_name + "\" and plotting " + str(num_plots) + " data files.")
    elif not os.path.exists(arguments[1]):
        print("Give a proper path name to the parent directory of the data files you want to use for plotting.")
        print("You gave \"" + str(arguments[1]) + "\", which is not a proper path name.")
        sys.exit(1)
    elif len(arguments[2]) == 0:
        print("Data file name can't be an empty string. The string you gave is \"" + arguments[2] + "\".")
        sys.exit(1)
    elif len(arguments[3]) == 0:
        print("Json file name can't be an empty string. The string you gave is \"" + arguments[3] + "\".")
        sys.exit(1)
    else:
        num_plots = arguments[2]
        data_file_name = arguments[3]
        json_file_name = arguments[4]
    
    fig = plt.figure()
    ax = fig.add_subplot(111) # nrows, ncols, index
    plot_relative_radius(ax, arguments[1], data_file_name, json_file_name, num_plots)

if __name__ == "__main__":
    main()