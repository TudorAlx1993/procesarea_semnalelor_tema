import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from laborator_8 import exercise_1
from statsmodels.tsa.arima.model import ARIMA


def main():
    output_dir = './outputs_homework'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    time, signal = exercise_1(1000, output_dir)
    exercise_2(time, signal, 0.5, output_dir)
    exercise_3(time, signal, 5, output_dir)
    exercise_4_arima_model(time, signal, 2, 5, output_dir)
    exercise_4_arima_best_params(time, signal, output_dir)


def exercise_3(time, signal, q, output_dir, seed=33):
    mean = 0.0
    std = 1.0
    np.random.seed(seed)
    errors = np.random.normal(loc=mean, scale=std, size=len(signal))

    target = pd.DataFrame(signal[(q + 1):])
    intercept = pd.DataFrame(np.ones_like(signal[(q + 1):]))

    data = pd.concat([target, intercept], axis=1)
    n = data.shape[0]
    for lag in range(q, 0, -1):
        regressor = pd.DataFrame(errors[lag:][:n])
        data = pd.concat([data, regressor], axis=1)

    y = data.iloc[:, 0].values
    Y = data.iloc[:, 1:].values
    x_star = np.dot(np.linalg.inv(np.dot(Y.transpose(), Y)), np.dot(Y.transpose(), y)).reshape(-1, 1)

    signal_fitted_v1 = np.dot(Y, x_star).reshape(-1)

    mse_v1 = calculate_mse(signal[(q + 1):], signal_fitted_v1)
    r_squared_v1 = calculate_r_squared(signal[(q + 1):], signal_fitted_v1)

    arima_model = ARIMA(signal, order=(0, 0, q))
    arima_model_fit = arima_model.fit()
    signal_fitted_v2 = arima_model_fit.fittedvalues

    mse_v2 = calculate_mse(signal, signal_fitted_v2)
    r_squared_v2 = calculate_r_squared(signal, signal_fitted_v2)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    axes[0].plot(time, signal, linewidth=3.0, label='Real data', color='red')
    axes[0].plot(time[(q + 1):], signal_fitted_v1, linewidth=3.0, label='Fitted data', color='blue')
    axes[0].set_title('My implementation of MA({})\nMSE={:.4f}\nR squared={:.2%}'.format(q, mse_v1, r_squared_v1))
    axes[1].plot(time, signal, linewidth=3.0, label='Real data', color='red')
    axes[1].plot(time, signal_fitted_v2, linewidth=3.0, label='Fitted data', color='blue')
    axes[1].set_title('MA({}) from statsmodels\nMSE={:.4f}\nR squared={:.2%}'.format(q, mse_v2, r_squared_v2))
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_3.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_3.pdf'), format='pdf')


def exercise_4_arima_model(time, signal, p, q, output_dir):
    if not (type(p) is int and type(q) is int and p >= 0 and q >= 0):
        raise ValueError('p and q parameters should be integers greater or equal to 0!')

    arima_model = ARIMA(signal, order=(p, 0, q))
    arima_model_fit = arima_model.fit()
    signal_fitted = arima_model_fit.fittedvalues

    mse = calculate_mse(signal, signal_fitted)
    r_squared = calculate_r_squared(signal, signal_fitted)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(time, signal, linewidth=3.0, label='Real data', color='red')
    ax.plot(time, signal_fitted, linewidth=3.0, label='Fitted data', color='blue')
    ax.set_title('ARMA({},{})\nMSE={:.4f}\nR squared={:.2%}'.format(p, q, mse, r_squared))
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_4_arma_model.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_4_arma_model.pdf'), format='pdf')


def exercise_4_arima_best_params(time, signal, output_dir, p_min=0, p_max=20, q_min=0, q_max=20):
    if not (type(p_min) is int and type(p_max) is int and p_min >= 0 and p_max > p_min):
        raise ValueError('parameters p_min and p_max should be integers with p_min>=0 and p_min<p_max!')

    if not (type(q_min) is int and type(q_max) is int and q_min >= 0 and q_max > q_min):
        raise ValueError('parameters q_min and q_max should be integers with q_min>=0 and q_min<q_max!')

    p_range = range(p_min, p_max + 1)
    q_range = range(q_min, q_max + 1)
    hyper_parameters = [(p, q) for p in p_range for q in q_range]

    hyper_parameters_tunning = []
    for p, q in hyper_parameters:
        arima_model = ARIMA(signal, order=(p, 0, q), trend=[1, 1, 0, 1])
        arima_model_fit = arima_model.fit()

        forecasted_signal = arima_model_fit.fittedvalues
        mse = calculate_mse(signal, forecasted_signal)
        aic = arima_model_fit.aic
        aicc = arima_model_fit.aicc
        bic = arima_model_fit.bic

        hyper_parameters_tunning.append(
            ((p, q), forecasted_signal, mse, aic, aicc, bic))

    file = open(os.path.join(output_dir, 'exercise_4_arima_best_params.txt'), 'w')
    file_content = 'P and Q hyperparameter tunning:\n'
    for (p, q), _, mse, aic, aicc, bic in hyper_parameters_tunning:
        file_content += '\t* p={} and q={}:\n'.format(p, q)
        file_content += '\t\t* MSE: {:.4f}\n'.format(mse)
        file_content += '\t\t* AIC: {:.4f}\n'.format(aic)
        file_content += '\t\t* AICC: {:.4f}\n'.format(aicc)
        file_content += '\t\t* BIC: {:.4f}\n'.format(bic)

    sorted_results_by_mse = sorted(hyper_parameters_tunning, key=lambda element: element[2])
    best_result_by_mse = sorted_results_by_mse[0]

    sorted_results_by_aic = sorted(hyper_parameters_tunning, key=lambda element: element[3])
    best_result_by_aic = sorted_results_by_aic[0]

    sorted_results_by_aicc = sorted(hyper_parameters_tunning, key=lambda element: element[4])
    best_result_by_aicc = sorted_results_by_aicc[0]

    sorted_results_by_bic = sorted(hyper_parameters_tunning, key=lambda element: element[5])
    best_result_by_bic = sorted_results_by_bic[0]

    file_content += '\n'

    results_for_plots = {'Min of MSE': best_result_by_mse,
                         'Min of AIC': best_result_by_aic,
                         'Min of AICC': best_result_by_aicc,
                         'Min of BIC:': best_result_by_bic}

    fig, axes = plt.subplots(nrows=len(results_for_plots), ncols=1, figsize=(10, 20))
    for ax, (criteria, results) in zip(axes, results_for_plots.items()):
        p, q = results[0]
        forecasted_signal = results[1]
        mse = results[2]
        aic = results[3]
        aicc = results[4]
        bic = results[5]

        file_content += 'Best hyperparameters using the folowwing metric: {}\n'.format(criteria)
        file_content += '\t* Best hyperparamters: p={} and q={}\n'.format(p, q)
        file_content += '\t\t* MSE: {:.4f}\n'.format(mse)
        file_content += '\t\t* AIC: {:.4f}\n'.format(aic)
        file_content += '\t\t* AICC: {:.4f}\n'.format(aicc)
        file_content += '\t\t* BIC: {:.4f}\n'.format(bic)

        ax.plot(time, signal, linewidth=3.0, color='red', label='Real data')
        ax.plot(time, forecasted_signal, linewidth=3.0, color='blue', label='Fitted data')
        ax.set_title(
            'ARIMA model hyperparameter tunning using the following metric: {}\nBest hyperparamters: p={},q={}\nMSE={:.4f}\nAIC={:.4f}\n,AICC={:.4f}\nBIC={:.4f}'.format(
                criteria, p, q, mse, aic, aicc, bic))
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_4_arima_best_params.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_4_arima_best_params.pdf'), format='pdf')

    file.write(file_content)
    file.close()


def calculate_mse(orignal_signal, predicted_signal):
    assert len(orignal_signal) == len(predicted_signal)

    mse = np.mean((orignal_signal - predicted_signal) ** 2)
    return mse


def exercise_2(time, signal_original, alpha_initial, output_dir):
    if not (type(alpha_initial) is float and alpha_initial >= 0.0 and alpha_initial <= 1.0):
        raise ValueError("parameter alpha_initial should be a float number between 0.0 and 1.0!")

    n = signal_original.shape[0]

    exp_smoothing_signal_initial_alpha_v1 = np.zeros(shape=n)
    exp_smoothing_signal_initial_alpha_v1[0] = signal_original[0]
    for time_step in range(1, n):
        exp_smoothing_signal_initial_alpha_v1[time_step] = alpha_initial * signal_original[time_step] + (
                1 - alpha_initial) * \
                                                           exp_smoothing_signal_initial_alpha_v1[time_step - 1]

    exp_smoothing_initial_alpha_signal_v2 = exponential_smoothing_of_signal(signal_original, alpha_initial)

    assert np.allclose(exp_smoothing_signal_initial_alpha_v1, exp_smoothing_initial_alpha_signal_v2)

    r_squared_initial_alpha = calculate_r_squared(signal_original, exp_smoothing_initial_alpha_signal_v2)
    sse_initial_alpha = np.sum((exp_smoothing_initial_alpha_signal_v2[:-1] - signal_original[1:]) ** 2)

    alpha_values = np.linspace(0.0, 1.0, 10 ** 2, True)
    exp_smoothing_alpha_values = [(alpha, exponential_smoothing_of_signal(signal_original, alpha)) for alpha in
                                  alpha_values]
    sse_alpha_values = [(alpha, np.sum((exp_smoothing_signal[:-1] - signal_original[1:]) ** 2)) for
                        alpha, exp_smoothing_signal in exp_smoothing_alpha_values]

    sorted_sse_alpha_values = sorted(sse_alpha_values, key=lambda element: element[1])
    best_result = sorted_sse_alpha_values[0]
    alpha_star = best_result[0]
    sse_alpha_star = best_result[1]

    exp_smoothing_alpha_star = exponential_smoothing_of_signal(signal_original, alpha_star)
    r_squared_alpha_star = calculate_r_squared(signal_original, exp_smoothing_alpha_star)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    axes[0].plot(time, signal_original, linewidth=3.0, label='Original signal', color='blue')
    axes[0].plot(time, exp_smoothing_signal_initial_alpha_v1, linewidth=3.0, label='Exponential smoothing', color='red')
    axes[0].set_title(
        'Original vs exponential smoothing with initial guess for alpha={}\nR squared={:.2%}\nSSE={:.4f}'.format(
            alpha_initial, r_squared_initial_alpha, sse_initial_alpha))
    axes[1].plot(time, signal_original, linewidth=3.0, label='Original signal', color='blue')
    axes[1].plot(time, exp_smoothing_alpha_star, linewidth=3.0, label='Exponential smoothing', color='red')
    axes[1].set_title(
        'Original vs exponential smoothing\nBest alpha={:.4f}\nR squared={:.2%}\nSSE={:.4f}'.format(alpha_star,
                                                                                                    r_squared_alpha_star,
                                                                                                    sse_alpha_star))
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_2.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_2.pdf'), format='pdf')


def calculate_r_squared(original_signal, predicted_signal):
    mean_original_signal = np.mean(original_signal)
    r_squared = 1 - np.sum((original_signal - predicted_signal) ** 2) / np.sum(
        (original_signal - mean_original_signal) ** 2)

    return r_squared


def exponential_smoothing_of_signal(signal, alpha):
    n = signal.shape[0]
    exp_smoothing_signal = np.zeros(shape=n)
    exp_smoothing_signal[0] = signal[0]
    exp_smoothing_signal[1:] = [alpha * np.sum(
        [(1 - alpha) ** (time_step - index) * signal[index] for index in range(1, time_step + 1)]) + (
                                        1 - alpha) ** time_step * signal[0] for time_step in
                                range(1, n)]

    return exp_smoothing_signal


if __name__ == '__main__':
    main()
