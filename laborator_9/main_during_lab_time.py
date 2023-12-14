import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_during_lab_time'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    time, signal = exercise_1(1000, output_dir)
    exercise_2(time, signal, 0.5, output_dir)


def exercise_2(time, signal_original, alpha_initial, output_dir):
    assert alpha_initial >= 0.0 and alpha_initial <= 1.0

    n = signal_original.shape[0]

    exp_smoothing_signal_v1 = np.zeros(shape=n)
    exp_smoothing_signal_v1[0] = signal_original[0]
    for time_step in range(1, n):
        exp_smoothing_signal_v1[time_step] = alpha_initial * signal_original[time_step] + (1 - alpha_initial) * \
                                             exp_smoothing_signal_v1[time_step - 1]

    exp_smoothing_signal_v2 = np.zeros(shape=n)
    exp_smoothing_signal_v2[0] = signal_original[0]
    exp_smoothing_signal_v2[1:] = [alpha_initial * np.sum(
        [(1 - alpha_initial) ** (time_step - index) * signal_original[index] for index in range(1, time_step + 1)]) + (
                                           1 - alpha_initial) ** time_step * signal_original[0] for time_step in
                                   range(1, n)]

    assert np.allclose(exp_smoothing_signal_v1, exp_smoothing_signal_v2)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    axes[0].plot(time, signal_original, linewidth=3.0, label='Original signal', color='blue')
    axes[0].plot(time, exp_smoothing_signal_v1, linewidth=3.0, label='Exponential smoothing', color='red')
    axes[0].set_title('Original vs exponential smoothing with initial guess for alpha={}'.format(alpha_initial))
    axes[0].legend()
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_2.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_2.pdf'), format='pdf')

    alpha_values=np.linspace(0.0,1.0,10**3,True)
    

# acest exercitiu l-am luat din laboratorul precedent
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
