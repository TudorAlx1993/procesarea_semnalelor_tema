import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_during_lab_time'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    signal = ex_1(output_dir, 1000)


def ex_1(output_dir, n):
    assert type(n) is int
    assert n > 0

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

    mean, sigma = 0.0, 0.4
    noise_component = np.random.normal(mean, sigma, size=n)

    computed_singal = trend_component + seasonal_component + noise_component

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
    axes[0].plot(time, computed_singal, linewidth=3.0)
    axes[0].set_title('Computed signal')
    axes[1].plot(time, trend_component, linewidth=3.0)
    axes[1].set_title('Trend component (second degree equation parameters: a={}, b={}, c={})'.format(a, b, c))
    axes[2].plot(time, seasonal_component, linewidth=3.0)
    axes[2].set_title('Seasonal component: $\omega_1={}$ Hz, $\omega_2={}$ Hz'.format(first_freq, second_freq))
    axes[3].plot(time, noise_component)
    axes[3].set_title('Gaussian noise with mean={} and std={}'.format(mean, sigma))
    for ax in axes:
        ax.set_xlabel('Time step')
        ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_1.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_1.pdf'), format='pdf')

    return computed_singal


if __name__ == '__main__':
    main()
