import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


def main():
    output_dir = './outputs_homework'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    time, signal = exercise_1(1000, output_dir)
    exercise_2(signal, output_dir)
    exercise_3(time, signal, 2, output_dir)
    exercise_4(signal, [1, 2, 3, 4, 5], [100, 150, 200, 225], output_dir)


def exercise_2(signal, output_dir):
    no_obs = signal.shape[0]
    no_lags = 1 + int(min(10 * np.log10(no_obs), no_obs - 1))

    acf_v1 = acf(x=signal, nlags=no_lags - 1)

    # pentru ACF folosesc formula de la acest link: https://otexts.com/fpp2/autocorrelation.html
    signal_mean = np.mean(signal)
    sst = np.sum((signal - signal_mean) ** 2)
    acf_v2 = np.array(
        [np.sum((signal[lag:] - signal_mean) * (signal[:(no_obs - lag)] - signal_mean)) / sst for lag in
         range(no_lags)])

    signal_without_mean = signal - signal_mean
    acf_v3 = np.correlate(a=signal_without_mean, v=signal_without_mean, mode='full')[(no_obs - 1):][:no_lags] / sst

    assert np.allclose(acf_v1, acf_v2)
    assert np.allclose(acf_v2, acf_v3)
    assert np.allclose(acf_v1, acf_v3)

    # in ceea ce priveste PACF exista mai multe metode de a o calcula
    # mai jos v2 se refera la implementarea proprie

    # varianta 1: pe baza ecuatiilor Yule-Walker
    pacf_yule_walker_v1 = pacf(x=signal, nlags=no_lags - 1, method='ywm')

    pacf_pacf_yule_walker_v2 = np.zeros(shape=no_lags)
    pacf_pacf_yule_walker_v2[0] = 1.0
    for lag in range(1, no_lags):
        acf_values = acf_v1[:lag]
        r_matrix = acf_v1[1:(lag + 1)].reshape(-1, 1)
        R_matrix = scipy.linalg.toeplitz(acf_values)
        theta_vector = np.dot(np.linalg.inv(R_matrix), r_matrix).flatten()
        pacf_pacf_yule_walker_v2[lag] = theta_vector[-1]

    assert np.allclose(pacf_yule_walker_v1, pacf_pacf_yule_walker_v2)

    # varianta 2: OLS
    pacf_ols_v1 = pacf(x=signal, nlags=no_lags - 1, method='ols')

    pacf_ols_v2 = np.zeros(shape=no_lags)
    pacf_ols_v2[0] = 1.0
    target = pd.DataFrame(signal)
    intercept = pd.DataFrame(np.ones(shape=no_obs))
    regressor = pd.DataFrame(signal)
    all_data = {}
    for lag_index in range(1, no_lags):
        if lag_index == 1:
            data = pd.concat([target, intercept, regressor.shift()], axis=1)
        else:
            data = pd.concat([data, data.iloc[:, -1].shift()], axis=1)

        all_data[lag_index] = data

    for lag_index in all_data.keys():
        data = all_data[lag_index]
        data = data.dropna()
        Y = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        x_star = np.dot(np.linalg.inv(np.dot(Y.transpose(), Y)), np.dot(Y.transpose(), y)).flatten()
        pacf_ols_v2[lag_index] = x_star[-1]

    assert np.allclose(pacf_ols_v1, pacf_ols_v2)

    # pentru plotare la PACF o sa folosesc varianta cu aproximarea Yull-Walker
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    axes[0].stem(range(no_lags), acf_v1)
    axes[0].set_title('Autocorrelation function (ACF)')
    axes[1].stem(range(no_lags), pacf_yule_walker_v1)
    axes[1].set_title('Partial autocorrelation function (PACF)')
    for ax in axes:
        ax.set_xlabel('Lag')
        ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_2.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_2.pdf'), format='pdf')


def exercise_3(time, signal, no_lags, output_dir):
    # graficul cu PACF ne arata ordinul modelului AR
    # din graficul PACF observam ca dupa lag=2 functia PACF scade exponential -> AR(2)

    target = pd.DataFrame(signal)
    intercept = pd.DataFrame(np.ones_like(signal))
    regressor = pd.DataFrame(signal)

    data = pd.concat([regressor.shift(index) for index in range(1, no_lags + 1)], axis=1)
    data = pd.concat([target, intercept, data], axis=1)
    data = data.dropna()

    y = data.iloc[:, 0].values
    Y = data.iloc[:, 1:].values

    x_star = np.dot(np.linalg.inv(np.dot(Y.transpose(), Y)), np.dot(Y.transpose(), y)).reshape(-1, 1)

    signal_fitted = np.dot(Y, x_star).reshape(-1)
    signal_original = np.copy(signal)

    no_skipped = len(signal) - len(signal_fitted)
    signal = signal[no_skipped:]
    signal_mean = np.mean(signal)
    r_squared = 1 - np.sum((signal - signal_fitted) ** 2) / np.sum((signal - signal_mean) ** 2)
    mse = np.mean((signal - signal_fitted) ** 2)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(time, signal_original, color='red', label='Original signal', linewidth=3.0)
    ax.plot(time[no_skipped:], signal_fitted, color='blue', label='Fitted signal', linewidth=3.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('Original vs fitted AR({}) model\nR squared={:.2%}\nMSE={:.5f}'.format(no_lags, r_squared, mse))
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'exercise_3.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_3.pdf'), format='pdf')


def exercise_4(signal, p_values, m_values, output_dir, no_folds=4):
    signal_folds = np.split(signal, no_folds)
    hyper_parameters = [(p, m) for p in p_values for m in m_values]

    results = []
    for p, m in hyper_parameters:
        mse_on_fold = []
        for signal in signal_folds:
            train_signal = signal[:m]
            test_signal = signal[m:]

            target = pd.DataFrame(train_signal)
            intercept = pd.DataFrame(np.ones_like(train_signal))
            regressor = pd.DataFrame(train_signal)

            data = pd.concat([regressor.shift(index) for index in range(1, p + 1)], axis=1)
            data = pd.concat([target, intercept, data], axis=1)
            data = data.dropna()

            y = data.iloc[:, 0].values
            Y = data.iloc[:, 1:].values

            x_star = np.dot(np.linalg.inv(np.dot(Y.transpose(), Y)), np.dot(Y.transpose(), y))

            data = np.concatenate(([1], test_signal[:p][::-1]))

            true_prediction = test_signal[p]
            predicted = np.sum(data * x_star)

            mse = np.mean((true_prediction - predicted) ** 2)
            mse_on_fold.append(mse)

        results.append((p, m, np.mean(mse_on_fold)))

    sorted_results_by_avg_of_mse = sorted(results, key=lambda x: x[2])
    best_p, best_m, min_avg_of_mse = sorted_results_by_avg_of_mse[0]

    file_content = 'Validation results using {} folds:\n'.format(no_folds)
    for p, m, avg_of_mse in results:
        file_content += '\t* p={}, m={}: average of MSE={:.5f}\n'.format(p, m, avg_of_mse)
    file_content += 'Best hyperparameters:\n'
    file_content += '\t* for p={} and m={} the average of MSE on folds is {:.5f}\n'.format(best_p, best_m,
                                                                                           min_avg_of_mse)

    file = open(os.path.join(output_dir, 'exercise_4.txt'), 'w')
    file.write(file_content)
    file.close()


def exercise_1(n, output_dir, seed=33):
    if not (type(n) is int and n > 0):
        raise ValueError('parameter n should be a strictly positive integer!')

    np.random.seed(seed)

    time = np.linspace(0, 1, num=n)

    a, b, c = 1.0, 1.5, 5.0
    trend_component = a * time ** 2 + b * time + c

    first_freq, second_freq = 10, 20
    amplitude = 1.0
    phase = 0.0
    generate_sinusoinal = lambda amplitude, time, freq, phase: amplitude * np.sin(2 * np.pi * time * freq + phase)
    seasonal_component = generate_sinusoinal(amplitude, time, first_freq, phase) + generate_sinusoinal(amplitude, time,
                                                                                                       second_freq,
                                                                                                       phase)

    mean, sigma = 0.0, 0.2
    noise_component = np.random.normal(mean, sigma, size=n)

    computed_signal = trend_component + seasonal_component + noise_component

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
    axes[0].plot(time, computed_signal, linewidth=3.0)
    axes[0].set_title('Computed signal')
    axes[1].plot(time, trend_component, linewidth=3.0)
    axes[1].set_title('Trend component (second degree equation parameters: a={}, b={}, c={})'.format(a, b, c))
    axes[2].plot(time, seasonal_component, linewidth=3.0)
    axes[2].set_title('Seasonal component: $\omega_1={}$ Hz, $\omega_2={}$ Hz'.format(first_freq, second_freq))
    axes[3].plot(time, noise_component)
    axes[3].set_title('Gaussian noise with mean={} and std={}'.format(mean, sigma))
    for ax in axes:
        ax.set_xlabel('Time step (s)')
        ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_1.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_1.pdf'), format='pdf')

    return time, computed_signal


if __name__ == '__main__':
    main()
