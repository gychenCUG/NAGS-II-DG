# 导入必要的模块和函数
import geatpy as ea
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from scipy import integrate
from scipy.stats import ks_2samp
from astropy.stats import kuiper_two, kuiper_false_positive_probability
import os

# Sundell and Saylor - 2017 - unmixing detrial geochronology age distributions

def transfer_toKDE(X, x):
    """ Kernel Density Estimation (KDE) function --- Gaussian kde . In this study,we use this function by default.

    :param X: age data
    :param x: age range and step size (KDE_min,KDE_max,KDE_Step)
    :return: KDEs
    """

    kde_list = []
    for i in range(len(X)):
        test1 = X.iloc[i, :]
        test1 = pd.to_numeric(test1, 'coerce')
        test1 = test1[test1 <= 4000]

        width = KDE_step
        zircon_list = test1.to_list()
        zircon_array = np.array(zircon_list)
        kde = gaussian_kde(zircon_array, bw_method=width / zircon_array.std())
        y = kde(x)
        kde_list.append(y)
    kde_list = np.array(kde_list)
    return kde_list

def transfer_toKDE2(X,x):
    """ Kernel Density Estimation (KDE) function --- Calculated directly from age data

    :param X:
    :param x:
    :return:
    """

    kde_list = []
    for i in range(len(X)):
        # 取出当前数据
        m = X.iloc[i, :]
        m = pd.to_numeric(m, 'coerce')
        m = m[m <= 4000]

        s = np.ones(len(m)) * KDE_step
        f = np.zeros((len(m), len(x)))

        for j in range(len(m)):
            f[j, :] = (1. / (s[j] * np.sqrt(2 * np.pi)) *
                       np.exp((-((x - m[j]) ** 2)) / (2 * (s[j] ** 2))) * KDE_step)
        kdeAi = (np.sum(f, axis=0) / len(m)).reshape(-1, 1)
        kde_list.append(kdeAi)
    kde_list = np.array(kde_list)
    kde_list = kde_list.reshape(len(X), len(x))
    return kde_list


def transfer_toCAD(X, x):
    """ cumulative distribution function (cdf_source)

    :param X: age data
    :param x: age range and step size (KDE_min,KDE_max,KDE_Step)
    :return: CDFs
    """

    cad_list = []
    for i in range(len(X)):
        test1 = X.iloc[i, :]
        test1 = pd.to_numeric(test1, 'coerce')
        test1 = test1[test1 <= 4000]

        bins = x
        hist, edges = np.histogram(test1, bins=bins, density=False)

        widths = np.diff(edges)
        cumulative = np.cumsum(hist * widths)

        # 计算累积百分比
        cumulative_pct = cumulative / cumulative[-1]

        cad_list.append(cumulative_pct)
    cad_list = np.array(cad_list)
    return cad_list


def compare_r2(kde_sink, kde_source):
    """ R2 metric calculation

    :param kde_sink: sink kde
    :param kde_source: source kde
    :return: R2 value
    """

    # Calculate the mean of kde_sink
    mean_kde_sink = np.mean(kde_sink)

    # Calculate the mean of kde
    mean_kde = np.mean(kde_source)

    # Calculate the difference between kde_sink data and the mean
    diff_kde_sink = kde_sink - mean_kde_sink

    # Calculate the difference between kde data and the mean
    diff_kde = kde_source - mean_kde

    # The numerator part of R2
    numerator = np.sum(diff_kde_sink * diff_kde)

    # The denominator part of R2
    denom_kde_sink = np.sum(diff_kde_sink ** 2)
    denom_kde = np.sum(diff_kde ** 2)
    denominator = np.sqrt(denom_kde_sink * denom_kde)

    # Calculate R2
    R2 = (numerator / denominator) ** 2

    # The relative inverse representation of R2
    # R2_n = 1 - R2

    return R2


def compare_D(cdf_sink, cdf_source):
    """ K-S metric calculation

    :param cdf_sink: sink cdf
    :param cdf_source: source cdf
    :return: D value
    """

    # Difference between sink area and source area samples
    diff_cdf_sink = cdf_sink - cdf_source

    # Difference between source area and sink area samples
    diff_cdf = cdf_source - cdf_sink

    # Maximum difference
    D = max(np.max(diff_cdf_sink), np.max(diff_cdf))

    return D


def compare_V(cdf_sink, cdf_source):
    """ Kuiper metric calculation

    :param cdf_sink: sink cdf
    :param cdf_source: source cdf
    :return: V value
    """

    # Difference between sink area and source area samples
    diff_cdf_sink = cdf_sink - cdf_source

    # Difference between source area and sink area samples
    diff_cdf = cdf_source - cdf_sink

    # Sum of maximum differences
    V = np.max(diff_cdf_sink) + np.max(diff_cdf)

    return V


# Calculate mean
def mean(nums):
    """ Calculate the mean of a list
    :param nums: list
    :return: mean
    """
    return sum(nums) / len(nums)


def normalize(nums):
    """
    Normalize numpy matrix
    :param nums:
    :return: Normalized proportional weight
    """
    normalized_result = []
    for i in range(len(nums)):
        nums_temp = nums[i]

        # 归一化
        normalize_nums = nums_temp / np.sum(nums_temp)

        # 标准化
        wghts = normalize_nums / np.sum(normalize_nums)

        normalized_result.append(wghts)
    normalized_result = np.array(normalized_result)
    return normalized_result

# ==========================Import data===========================
# Import Data
# Source S1,S2,...,Sn
# GOBI、YR、NTP、CNN、CSD、QB、NQP

# Sink T1,T2,...,T12
# JX、LC、LD、BGY、XF、XN、LNT、JB、ZTS、HMG、LT、WN

KDE_min = 0  # min
KDE_max = 4000  # max
KDE_step = 20  # step

x = np.arange(KDE_min, KDE_max + KDE_step, KDE_step)

data = pd.read_excel('../ggge21358-sup-0003-2016gc006774-ds02-extended.xlsx')

ages_all = data.iloc[:,::2]

ages_all = ages_all.T

# Retrieve source area data
sink_data = ages_all.iloc[0,:]
sink_data = sink_data.values.reshape(1,-1)
sink_data = pd.DataFrame(sink_data)

source_1 = ages_all.iloc[1,:]
source_1 = pd.to_numeric(source_1, 'coerce')
source_1 = source_1[source_1 <= 4000]

source_2 = ages_all.iloc[2,:]
source_2 = pd.to_numeric(source_2, 'coerce')
source_2 = source_2[source_2 <= 4000]

source_3 = ages_all.iloc[3,:]
source_3 = pd.to_numeric(source_3, 'coerce')
source_3 = source_3[source_3 <= 4000]

source_4 = ages_all.iloc[4,:]
source_4 = pd.to_numeric(source_4, 'coerce')
source_4 = source_4[source_4 <= 4000]

source_5 = ages_all.iloc[5,:]
source_5 = pd.to_numeric(source_5, 'coerce')
source_5 = source_5[source_5 <= 4000]

source_6 = ages_all.iloc[6,:]
source_6 = pd.to_numeric(source_6, 'coerce')
source_6 = source_6[source_6 <= 4000]

source_7 = ages_all.iloc[7,:]
source_7 = pd.to_numeric(source_7, 'coerce')
source_7 = source_7[source_7 <= 4000]

source_8 = ages_all.iloc[8,:]
source_8 = pd.to_numeric(source_8, 'coerce')
source_8 = source_8[source_8 <= 4000]

source_9 = ages_all.iloc[9,:]
source_9 = pd.to_numeric(source_9, 'coerce')
source_9 = source_9[source_9 <= 4000]

source_10 = ages_all.iloc[10,:]
source_10 = pd.to_numeric(source_10, 'coerce')
source_10 = source_10[source_10 <= 4000]

# form the required distribution
# sink KDE (1,200)
targetkde = transfer_toKDE(sink_data, x)
# sink CAD (1,200)
targetcad = transfer_toCAD(sink_data, x)

# The total number of grains subsampled
num_grains = 1000

# The number of times to sample under the proportion: licht_N
Cepoch = 2




def process_sample(ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6, ratio_7, ratio_8, ratio_9, ratio_10):
    """ Generate new KDE and CAD curves under the number of grains pumped.
    The input parameter is the number of grains, and the output is the KDE curve、CDF curve.
    Calculate the indicators under Cepoch times of reorganization.

    :param ratio_1: contribution1
    :param ratio_2: contribution2
    :param ratio_3: contribution3
    :param ratio_4: contribution4
    :param ratio_5: contribution5
    :param ratio_6: contribution6
    :param ratio_7: contribution7
    :param ratio_8: contribution8
    :param ratio_9: contribution9
    :param ratio_10: contribution10
    :return: r2_mean 、v_mean 、d_mean
    """

    wghts = [ratio_1, ratio_2, ratio_3, ratio_4, ratio_5, ratio_6, ratio_7, ratio_8, ratio_9, ratio_10]

    # Number of grains extracted
    num_Ages = np.round(np.array(wghts) * num_grains).astype(int)

    # Dealing with grains difference issues
    sum_Ages = np.sum(num_Ages)  # Sum of weights after rounding
    diff = sum_Ages - num_grains  # The difference from the total number of grains
    if diff > 0:
        max_idx = np.argmax(num_Ages)
        num_Ages[max_idx] -= diff  # Subtract the difference at the maximum weight
    elif diff < 0:
        min_idx = np.argmin(num_Ages)
        num_Ages[min_idx] += abs(diff)  # Add the difference at the minimum weight

    R2Results = []
    VmaxResults = []
    DmaxResults = []

    for j in range(Cepoch):
        # 1、First do zircon mixing to form the current distribution
        samples = []

        # subsampling
        source_1_sam = source_1.sample(num_Ages[0], replace=False)
        source_2_sam = source_2.sample(num_Ages[1], replace=False)
        source_3_sam = source_3.sample(num_Ages[2], replace=False)
        source_4_sam = source_4.sample(num_Ages[3], replace=False)
        source_5_sam = source_5.sample(num_Ages[4], replace=False)
        source_6_sam = source_6.sample(num_Ages[5], replace=False)
        source_7_sam = source_7.sample(num_Ages[6], replace=False)
        source_8_sam = source_8.sample(num_Ages[7], replace=False)
        source_9_sam = source_9.sample(num_Ages[8], replace=False)
        source_10_sam = source_10.sample(num_Ages[9], replace=False)

        # combination
        samples.append(
            pd.concat([source_1_sam, source_2_sam, source_3_sam, source_4_sam, source_5_sam, source_6_sam, source_7_sam,
                       source_8_sam, source_9_sam, source_10_sam], ignore_index=True))
        samples = samples[0].T
        samples = pd.DataFrame({'pot': samples})
        samples = samples.T

        # 2、Then calculate kde and cdf

        # 1 KDE (1,200)
        kde = transfer_toKDE(samples,x)

        # 1 CAD (1,200)
        cad = transfer_toCAD(samples,x)

        # Calculate R2, Vmax, and Dmax at the j-th epoch under the current ratio
        r2 = compare_r2(kde[0], targetkde[0])
        vmax = compare_V(cad[0], targetcad[0])
        dmax = compare_D(cad[0], targetcad[0])

        R2Results.append(r2)
        DmaxResults.append(dmax)
        VmaxResults.append(vmax)

    R2_mean = mean(R2Results)
    Dmax_mean = mean(DmaxResults)
    Vmax_mean = mean(VmaxResults)

    return R2_mean, Vmax_mean, Dmax_mean


# Define the problem class that needs to be optimized
class MyProblem(ea.Problem):
    # Initialization problem
    def __init__(self):
        name = 'MyProblem'  # Initialize question name
        M = 3  # Initialize target dimensions
        maxormins = [-1, 1, 1]  # Initialize the maximization and minimization marks of each goal, 1 means minimization, -1 means maximization R2, Vmax, Dmax
        Dim = 10  # Initialize the number of decision variables
        varTypes = [0] * Dim  # Initialize the decision variable type, 0 means continuous variable, 1 means discrete (type of decision variable, 0: real number; 1: integer)
        lb = [0] * Dim  # decision variable lower bound
        ub = [1] * Dim  # decision variable upper bound
        lbin = [1] * Dim  # Does the lower boundary of the decision variable contain
        ubin = [1] * Dim  # Does the upper boundary of the decision variable contain
        # Call the parent class constructor to complete instantiation
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def aimFunc(self, pop):
        # Get the decision variable matrix, which is equal to the phenotype matrix Phen of the population
        Vars = pop.Phen

        # Normalized decision variables
        Vars = normalize(Vars)

        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = Vars[:, [4]]
        x6 = Vars[:, [5]]
        x7 = Vars[:, [6]]
        x8 = Vars[:, [7]]
        x9 = Vars[:, [8]]
        x10 = Vars[:, [9]]

        # save r2 [Nind,1]
        r2 = np.zeros((len(x1), 1))

        # save dmax [Nind,1]
        v = np.zeros((len(x1), 1))

        # save dmax [Nind,1]
        d = np.zeros((len(x1), 1))

        # Get the decision variables of the current individual
        for i in range(len(x1)):
            t_x1 = x1[i][0]
            t_x2 = x2[i][0]
            t_x3 = x3[i][0]
            t_x4 = x4[i][0]
            t_x5 = x5[i][0]
            t_x6 = x6[i][0]
            t_x7 = x7[i][0]
            t_x8 = x8[i][0]
            t_x9 = x9[i][0]
            t_x10 = x10[i][0]

            # Record three metrics of the current individual
            r2[i][0],v[i][0],d[i][0] = process_sample(t_x1, t_x2, t_x3, t_x4, t_x5, t_x6, t_x7, t_x8, t_x9, t_x10)

        # record target value
        pop.ObjV = np.hstack([r2,v,d])





