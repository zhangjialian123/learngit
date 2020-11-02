import numpy as np
import os
from scipy.fftpack import fft, ifft
import pywt
from scipy.signal import hilbert
import joblib
import sklearn


def RF_analysis(points, fs, L):  # 单个文件保存的点数，采样频率，管长
    class VMD:
        def __init__(self, K, alpha, tau, tol=1e-7, maxIters=200, eps=1e-9):
            """
            :param K: 模态数
            :param alpha: 每个模态初始中心约束强度
            :param tau: 对偶项的梯度下降学习率
            :param tol: 终止阈值
            :param maxIters: 最大迭代次数
            :param eps: eps
            """
            self.K = K
            self.alpha = alpha
            self.tau = tau
            self.tol = tol
            self.maxIters = maxIters
            self.eps = eps

        def __call__(self, f):
            N = f.shape[0]
            # 对称拼接
            f = np.concatenate((f[:N // 2][::-1], f, f[N // 2:][::-1]))
            T = f.shape[0]
            t = np.linspace(1, T, T) / T
            omega = t - 1. / T
            # 转换为解析信号
            f = hilbert(f)
            f_hat = np.fft.fft(f)
            u_hat = np.zeros((self.K, T), dtype=np.complex)
            omega_K = np.zeros((self.K,))
            lambda_hat = np.zeros((T,), dtype=np.complex)
            # 用以判断
            u_hat_pre = np.zeros((self.K, T), dtype=np.complex)
            u_D = self.tol + self.eps

            # 迭代
            n = 0
            while n < self.maxIters and u_D > self.tol:
                for k in range(self.K):
                    # u_hat
                    sum_u_hat = np.sum(u_hat, axis=0) - u_hat[k, :]
                    res = f_hat - sum_u_hat
                    u_hat[k, :] = (res + lambda_hat / 2) / \
                                  (1 + self.alpha * (omega - omega_K[k]) ** 2)

                    # omega
                    u_hat_k_2 = np.abs(u_hat[k, :]) ** 2
                    omega_K[k] = np.sum(omega * u_hat_k_2) / np.sum(u_hat_k_2)

                # lambda_hat
                sum_u_hat = np.sum(u_hat, axis=0)
                res = f_hat - sum_u_hat
                lambda_hat -= self.tau * res

                n += 1
                u_D = np.sum(np.abs(u_hat - u_hat_pre) ** 2)
                u_hat_pre[::] = u_hat[::]

            # 重构，反傅立叶之后取实部
            u = np.real(np.fft.ifft(u_hat, axis=-1))
            u = u[:, N // 2: N // 2 + N]

            omega_K = omega_K * T / 2
            idx = np.argsort(omega_K)
            omega_K = omega_K[idx]
            u = u[idx, :]
            return u, omega_K

    # file_name=2

    def file_all_data(file_path):
        file_list = os.listdir(file_path)
        file_list11 = [[]] * len(file_list)
        f2 = [[]] * len(file_list)
        for i in range(0, len(file_list)):
            file_list11[i] = os.path.join(file_path, file_list[i])
            with open(file_list11[i], "r") as f1:
                f2[i] = f1.read()
                f1.close()
        return f2

    ###############################################################################
    ####-------------------------  信号损伤指数的计算
    ####------------------------   有量纲特征值
    ###-----------------------  指标1， 均值

    def mean_value(data, points):
        all_sum = np.sum(data)
        average = all_sum / points
        return average

    ###-----------------------  指标2， 均方根
    def root_mean_square(data, points):
        all_sum = np.sum(np.square(data))
        average = all_sum / points
        return np.sqrt(average)

    ###-----------------------  指标3， 方差
    def var_value(data, points):
        vv = np.var(data)
        return vv

    ###-----------------------   指标4， 峰值
    def max_value(data, points):
        mv = max(data)
        return mv

    ###-----------------------  指标5， 波形指标
    def shape_factor(data, points):
        r_t_s = root_mean_square(data, points)
        m_v = mean_value(np.abs(data), points)
        sf = r_t_s / m_v
        return sf

    ###-----------------------  指标6， 峰值指标
    def crest_factor(data, points):
        r_t_s = root_mean_square(data, points)
        cf = np.max(data) / r_t_s
        return cf

    ###-----------------------  指标7， 脉冲指标
    def impulse_factor(data, points):
        m_v = mean_value(np.abs(data), points)
        If = np.max(data) / m_v
        return If

    ###-----------------------  指标8， 裕度指标
    def clearance_factor(data, points):
        a = np.sum(np.sqrt(np.abs(data)))
        clf = np.max(data) / np.square(a / points)
        return clf

    ###-----------------------  指标9， 峭度指标
    def kurtosis_value(data, points):
        m_v = mean_value(data, points)
        a = [i - m_v for i in data]
        b = [i ** 4 for i in a]
        c = [i * i for i in a]
        kv = points * np.sum(b) / np.square(np.sum(c))
        return kv

    def Energy_value(data, points):
        EV = np.sum(np.square(data))
        return EV

    ###-----------------------  构造指标向量

    def shape_factor_set(data, points):
        sig_mean = mean_value(data, points)
        sig_rms = root_mean_square(data, points)
        sig_var = var_value(data, points)
        sig_max = max_value(data, points)
        sig_sf = shape_factor(data, points)
        sig_cf = crest_factor(data, points)
        sig_if = impulse_factor(data, points)
        sig_clf = clearance_factor(data, points)
        sig_ka = kurtosis_value(data, points)
        sig_ev = Energy_value(data, points)
        set = [sig_mean, sig_rms, sig_var, sig_max, sig_sf,
               sig_cf, sig_if, sig_clf, sig_ka, sig_ev]
        return set

    epsn = 1e-8

    def fft_mean(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        return np.mean(freq_spectrum)

    def fft_var(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        return np.var(freq_spectrum)

    def fft_std(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        return np.std(freq_spectrum)

    def fft_entropy(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        pr_freq = freq_spectrum * 1.0 / _freq_sum_

        entropy = -1 * np.sum([np.log2(p + 1e-5) * p for p in pr_freq])

        return entropy

    def fft_energy(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        return np.sum(freq_spectrum ** 2) / len(freq_spectrum)

    def fft_skew(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        _fft_mean, _fft_std = fft_mean(sequence_data), fft_std(sequence_data)

        return np.mean([0 if _fft_std < epsn else np.power((x - _fft_mean) / _fft_std, 3)

                        for x in freq_spectrum])

    def fft_kurt(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        _fft_mean, _fft_std = fft_mean(sequence_data), fft_std(sequence_data)

        return np.mean([0 if _fft_std < epsn else np.power((x - _fft_mean) / _fft_std, 4)

                        for x in freq_spectrum])

    def fft_shape_mean(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        shape_sum = np.sum([x * freq_spectrum[x]

                            for x in range(len(freq_spectrum))])

        return 0 if _freq_sum_ < epsn else shape_sum * 1.0 / _freq_sum_

    def fft_shape_std(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        shape_mean = fft_shape_mean(sequence_data)

        var = np.sum([0 if _freq_sum_ < epsn else np.power((x - shape_mean), 2) * freq_spectrum[x]

                      for x in range(len(freq_spectrum))]) / _freq_sum_

        return np.sqrt(var)

    def fft_shape_skew(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        shape_mean = fft_shape_mean(sequence_data)

        return np.sum([np.power((x - shape_mean), 3) * freq_spectrum[x]

                       for x in range(len(freq_spectrum))]) / _freq_sum_

    def fft_shape_kurt(sequence_data):
        def fft_fft(sequence_data):
            fft_trans = np.abs(np.fft.fft(sequence_data))

            freq_spectrum = fft_trans[1:int(
                np.floor(len(sequence_data) * 1.0 / 2)) + 1]

            _freq_sum_ = np.sum(freq_spectrum)

            return freq_spectrum, _freq_sum_

        freq_spectrum, _freq_sum_ = fft_fft(sequence_data)

        shape_mean = fft_shape_mean(sequence_data)

        return np.sum([np.power((x - shape_mean), 4) * freq_spectrum[x] - 3

                       for x in range(len(freq_spectrum))]) / _freq_sum_

    def frequency_factor_set(sequence_data):
        Z1 = fft_mean(sequence_data)
        Z2 = fft_var(sequence_data)
        Z3 = fft_std(sequence_data)
        Z4 = fft_entropy(sequence_data)
        Z5 = fft_energy(sequence_data)
        Z6 = fft_skew(sequence_data)
        Z7 = fft_kurt(sequence_data)
        Z8 = fft_shape_mean(sequence_data)
        Z9 = fft_shape_std(sequence_data)
        Z10 = fft_shape_skew(sequence_data)
        Z11 = fft_shape_kurt(sequence_data)
        Z = [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11]
        return Z

    ####-------------------------------------------构建多维向量
    ###------------------------------- 波形输入

    def shape_input_set(f, points):
        set = []
        for i in range(len(f)):
            data_compress = f[i]
            single_input = shape_factor_set(data_compress, points)
            set.append(single_input)
        return set

    ###------------------------------- 频域输入

    def fre_input_set(f, points):
        set = []
        for i in range(len(f)):
            data_compress = f[i]
            fre_distribution = frequency_factor_set(data_compress)
            set.append(fre_distribution)
        return set

    ###-----------------------------  时域特征+频域特征合集

    def all_input_set(f, points):
        set = []
        for i in range(len(f)):
            data_compress = np.asarray(f[i]).tolist()[0]
            fre_distribution = frequency_factor_set(data_compress)
            single_input = shape_factor_set(data_compress, points)
            all_input = single_input + fre_distribution
            set.append(all_input)
        return set

    #####vmd分解+小波去噪

    def wdt(f, points):
        set = []
        kk = 4
        w = pywt.Wavelet('db10')
        # for i in range(len(f)):
        data = f.split('\n')
        #data=np.asarray(data)
        NN = int(points)
        data_compress = [[]] * NN
        for j in range(0, NN):
            a = float(data[j]) / 10000
            data_compress[j] = round(a, 4)
        data_vmd, omega_k = vmd(np.asarray(data_compress))
        for d in range(kk - 1):
            data_vmd_buff = np.zeros(shape=(1, points))
            maxlev = pywt.dwt_max_level(len(data_vmd[d, :]), w.dec_len)
            threshold = 0.15
            coff = pywt.wavedec(data_vmd[d, :], 'db10', level=maxlev)
            for c in range(1, len(coff)):
                coff[c] = pywt.threshold(coff[c], threshold * max(coff[c]))
                data_vmd[d, :] = pywt.waverec(coff, 'db10')
            data_vmd_buff = data_vmd[d, :] + data_vmd_buff
        set.append(data_vmd_buff)
        return set

    def pre(file_path, modelpath):

        f_list = os.listdir(file_path)
        f_length = len(f_list)
        f_list.sort(key=lambda fn: os.path.getmtime(file_path + '\\' + fn))
        f_all = file_all_data(file_path)
        f_single_path = os.path.join(file_path, f_list[-1])
        with open(f_single_path, "r") as f:
            f_single = f.read()
            f.close()

        f_wdt = wdt(f_single, points)
        f_test = all_input_set(f_wdt, points)

        model = joblib.load(modelpath)
        result = model.predict(np.array(f_test).reshape(1, -1))
        return result, f_length, f_wdt, f_all

    def corr(f1, f3, points):
        a = np.asarray(f1)
        b = np.asarray(f3)
        a = np.squeeze(a)
        b = np.squeeze(b)
        # loc=signal.fftconvolve(a-a.mean(),b-b.mean(),'full')
        loc = np.correlate(a, b, 'full')
        loc = np.argmax(loc) + 1 - points
        return loc

    KK = 3
    alpha = 2000
    tau = 1e-6
    vmd = VMD(KK, alpha, tau)
    modelpath = r'C:\Users\1\Desktop\Pipeline leakage monitoring\V1.51-2020-8-10\References\model.pkl'
    file_path1 = r'C:\Users\1\Desktop\Pipeline leakage monitoring\V1.51-2020-8-10\Data\1'
    file_path2 = r'C:\Users\1\Desktop\Pipeline leakage monitoring\V1.51-2020-8-10\Data\2'
    file_path3 = r'C:\Users\1\Desktop\Pipeline leakage monitoring\V1.51-2020-8-10\Data\3'
    result1, f1_length, f1, f1_all = pre(file_path1, modelpath)
    result2, f2_length, f2, f2_all = pre(file_path2, modelpath)
    result3, f2_length, f3, f3_all = pre(file_path3, modelpath)

    result = [result1, result2, result3]
    v = 1480
    count = 0
    for i in range(3):
        if result[i] == 1:
            count = count + 1
        else:
            count = count

    if count % 2 == 0 or count == 3:
        signal_leak = True
        location = corr(f1, f3, points)
        locat = (v * location / 500 + L) / 2
    else:
        signal_leak = False
        locat = 0
    #signal_leak("utf8","ignore")
    #locat("utf8", "ignore")
    return (signal_leak,locat)


'''points = 200000
fs = 500
L = 100
x= RF_analysis(points, fs, L)
print(x)
print(type(x))'''


