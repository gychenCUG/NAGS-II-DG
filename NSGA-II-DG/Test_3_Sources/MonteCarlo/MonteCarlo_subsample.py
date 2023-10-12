import numpy as np
import pandas as pd
import random
from scipy.stats import gaussian_kde


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

    :param X: age data
    :param x: age range and step size (KDE_min,KDE_max,KDE_Step)
    :return: KDEs
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


def generate_ratio(n):
    """ Generate proportion based on the number of source areas

    :param n: the number of source areas
    :return: weight ratio
    """

    N = n

    # Generate an array of integers from 1 to N-1
    # x3 = np.arange(1, N)
    x3 = np.arange(N - 1)

    # Randomly permute the array to get a random index array for generating weights later
    samples = np.random.permutation(x3)

    # Generate an array of integers from 1 to N
    # x3 = np.arange(1, N+1)
    x3 = np.arange(N)

    # Compute the default weight value for each sample
    m = 1 / N

    # Initialize an array of length 1 and element 0
    sampled_y = np.zeros((1, N + 1))

    # Set the last element of the array to 1
    sampled_y[0, N] = 1

    # Calculate the right endpoint position of each weight segment corresponding to a random index
    y_line = (samples + 1) * m

    # print(samples)
    # Iterate over each weight segment corresponding to a random index
    for i in range(N - 1):
        # Compute the maximum weight value on the left of the segment
        # if samples[i] == 0:
        #     tmp_min = sampled_y[0,0]
        # else:
        tmp_min = np.max(sampled_y[0, :samples[i] + 1])

        # Compute the minimum weight value on the right of the segment
        zerosindex = np.nonzero(sampled_y[0, samples[i] + 1:])
        # print(zerosindex)
        index1 = zerosindex[0] + samples[i] + 1
        list_tmp = []
        for i1 in index1:
            # print(i)
            # print(sampled_y[0,i])
            list_tmp.append(sampled_y[0, i1])
            # print(list_tmp)

        tmp_max = np.min(np.array(list_tmp))
        # tmp_max = np.min(np.nonzero(sampled_y[0, samples[i]+1:]))

        # If the maximum weight value is less than the default weight value of the right endpoint,
        # set the right endpoint to a random value within the segment
        if tmp_min < y_line[i]:
            sampled_y[0, samples[i] + 1] = y_line[i] + random.random() * (tmp_max - y_line[i])
        # Otherwise, set the right endpoint to a random value between existing weight values
        else:
            sampled_y[0, samples[i] + 1] = tmp_min + random.random() * (tmp_max - tmp_min)

    # Compute the random weight value for each sample
    wghts_tmp = np.diff(sampled_y)

    # Randomly permute the array of weight values to get a random weight vector
    wghts = np.random.permutation(wghts_tmp[0])

    wghts = wghts.flatten()

    return wghts


# 一个模型中的第i个样本，计算重复后的平均指标
def process_sample(i, source_props, R2, V, D):
    """ Calculate the average quantitative metrcis value under the current proportional weight

    :param i: current number of repetitions
    :param source_props: weight ratio list
    :param R2: R2 metric calculation
    :param V: V metric calculation
    :param D: D metric calculation
    """

    # Generation ratio
    temp = generate_ratio(3)

    # Save ratio
    source_props[i] = temp

    # Number of grains extracted
    temp = np.round(np.array(temp) * num_grains).astype(int)

    # Dealing with grains difference issues
    sum_Ages = np.sum(temp)  # Sum of weights after rounding
    diff = sum_Ages - num_grains  # The difference from the total number of grains
    if diff > 0:
        max_idx = np.argmax(temp)
        temp[max_idx] -= diff  # Subtract the difference at the maximum weight
    elif diff < 0:
        min_idx = np.argmin(temp)
        temp[min_idx] += abs(diff)  # Add the difference at the minimum weight

    R2Results = []
    VmaxResults = []
    DmaxResults = []

    for j in range(Cepoch):
        # 1、First do zircon mixing to form the current distribution
        samples = []

        # subsampling
        source_1_sam = source_1.sample(temp[0], replace=False)
        source_2_sam = source_2.sample(temp[1], replace=False)
        source_3_sam = source_3.sample(temp[2], replace=False)

        # combination
        samples.append(
            pd.concat([source_1_sam, source_2_sam, source_3_sam], ignore_index=True))
        samples = samples[0].T
        samples = pd.DataFrame({'pot': samples})
        samples = samples.T

        # 2、Then calculate kde and cdf
        # 1 KDE (1,200)
        kde = transfer_toKDE(samples, x)

        # 1 CAD (1,200)
        cad = transfer_toCAD(samples, x)

        # Calculate R2, Vmax, and Dmax at the j-th epoch under the current ratio
        r2 = compare_r2(kde[0], targetkde[0])
        vmax = compare_V(cad[0], targetcad[0])
        dmax = compare_D(cad[0], targetcad[0])

        R2Results.append(r2)
        VmaxResults.append(vmax)
        DmaxResults.append(dmax)

    # Calculate the average of each metric and save it
    R2[i] = mean(R2Results)
    V[i] = mean(VmaxResults)
    D[i] = mean(DmaxResults)


def save(model_num, source_props, R2, V, D):
    """ Keep records

    :param model_num: Number of model runs
    :param source_props: weight ratio list
    :param R2: R2 metric list
    :param V: V metric list
    :param D: D metric list
    """


    # Arrange them according to the size of the indicators from small to large, retaining the first num_keep optimal ratios and their indicator values
    df = pd.DataFrame(source_props, columns=[f'source_{i + 1}' for i in range(num_sources)])
    df['r2'], df['v'], df['d'] = R2, V, D

    # Save all data
    df.to_csv(f'model_{model_num}_sink_all.csv', index=False)

    # Sort by three metrics
    df_r2 = df.sort_values(by=['r2'], ascending=[False])
    df_v = df.sort_values(by=['v'], ascending=[True])
    df_d = df.sort_values(by=['d'], ascending=[True])

    # Save header data
    df_r2_head = df_r2.head(num_keep)
    df_v_head = df_v.head(num_keep)
    df_d_head = df_d.head(num_keep)

    # Save results
    df_r2_head.to_csv(f'model_{model_num}_sink_r2head.csv', index=False)
    df_v_head.to_csv(f'model_{model_num}_sink_vhead.csv', index=False)
    df_d_head.to_csv(f'model_{model_num}_sink_dhead.csv', index=False)


if __name__ == '__main__':
    # ==========================Import data===========================
    # Import Data
    # Source S1,S2,...,Sn

    # Sink T1,T2,...,T12

    KDE_min = 0  # min
    KDE_max = 4000  # max
    KDE_step = 20  # step

    x = np.arange(KDE_min, KDE_max + KDE_step, KDE_step)

    data = pd.read_excel('../ggge21358-sup-0002-2016gc006774-ds01-extended-181.xlsx')

    ages_all = data.iloc[:, ::2]

    ages_all = ages_all.T

    # Retrieve source area data
    sink_data = ages_all.iloc[0, :]
    sink_data = sink_data.values.reshape(1, -1)
    sink_data = pd.DataFrame(sink_data)

    source_1 = ages_all.iloc[1, :]
    source_1 = pd.to_numeric(source_1, 'coerce')
    source_1 = source_1[source_1 <= 4000]

    source_2 = ages_all.iloc[2, :]
    source_2 = pd.to_numeric(source_2, 'coerce')
    source_2 = source_2[source_2 <= 4000]

    source_3 = ages_all.iloc[3, :]
    source_3 = pd.to_numeric(source_3, 'coerce')
    source_3 = source_3[source_3 <= 4000]

    # form the required distribution
    # sink KDE (1,200)
    targetkde = transfer_toKDE(sink_data, x)
    # sink CAD (1,200)
    targetcad = transfer_toCAD(sink_data, x)

    # Number of Monte Carlo model runs
    model_num = 1

    # Number of source areas
    num_sources = 3

    # The number of times the ratio was generated
    num_samples = 20

    # Number of heads reserved
    num_keep = 10

    # The number of times to sample under the proportion: licht_N
    Cepoch = 2

    # The number of grains subsampled
    num_grains = 1000


    for i in range(model_num):
        # Storage ratio under a single model
        source_props = np.zeros((num_samples, 3))

        # Calculate three quantitative similarity metrics at each ratio
        R2, V, D = np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples)

        for j in range(num_samples):
            process_sample(j, source_props, R2, V, D)

        # save results
        save(i, source_props, R2, V, D)
