import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


def main():
    output_dir = './outputs_tema'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exercitiul_1(output_dir)
    exercitiul_2(output_dir)
    exercitiul_8(output_dir)


def exercitiul_8(output_dir):
    start, stop = -np.pi / 2, np.pi / 2
    no_points = 10 ** 4
    alpha = np.linspace(start=start, stop=stop, num=no_points)

    sinus_real = np.sin(alpha)
    sinus_taylor = alpha
    sinus_pade = (alpha - 7 * alpha ** 3 / 60) / (1 + alpha ** 2 / 20)

    error_sinus_real_vs_taylor = sinus_real - sinus_taylor
    error_sinus_real_vs_pade = sinus_real - sinus_pade

    fig, (first_ax, second_ax) = plt.subplots(nrows=1, ncols=2, figsize=(25, 7))
    first_ax.plot(alpha, sinus_real, color='red', label='Real sinus', linewidth=4.5)
    first_ax.plot(alpha, sinus_taylor, color='blue', label='Taylor sinus', linewidth=4.5)
    first_ax.plot(alpha, sinus_pade, color='green', label='Pade sinus', linewidth=4.5)
    first_ax.set_xlabel('$Alpha$')
    first_ax.set_ylabel('$Sinus$')
    first_ax.set_title('Real, Taylor and Pade sinus')
    first_ax.legend(loc='upper left')
    second_ax.plot(alpha, error_sinus_real_vs_taylor, color='yellow', label='Error: real vs Taylor sinus',
                   linewidth=4.5)
    second_ax.plot(alpha, error_sinus_real_vs_pade, color='purple', label='Error: real vs Pade sinus', linewidth=4.5)
    second_ax.set_xlabel('$Alpha$')
    second_ax.set_ylabel('$Error$')
    second_ax.set_title('Taylor and Pade sinus errors')
    second_ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_3.png'))

    # se va afisa doar pentru alpha>0 deoarece pe intervalul [-pi/2,0) sinusul este negativ, iar sin(0)=0 si deci nu pot aplica functia logaritm
    fig, (first_ax, second_ax) = plt.subplots(nrows=1, ncols=2, figsize=(25, 7))
    first_ax.plot(alpha, sinus_real, color='red', label='Real sinus', linewidth=4.5)
    first_ax.plot(alpha, sinus_taylor, color='blue', label='Taylor sinus', linewidth=4.5)
    first_ax.plot(alpha, sinus_pade, color='green', label='Pade sinus', linewidth=4.5)
    first_ax.set_yscale("log", base=10)
    first_ax.set_xlabel('$Alpha$')
    first_ax.set_ylabel('$Sinus$')
    first_ax.set_title('Real, Taylor and Pade sinus (OY ax is on logarithmic scale)')
    first_ax.legend(loc='upper left')
    second_ax.plot(alpha, error_sinus_real_vs_taylor, color='yellow', label='Error: real vs Taylor sinus',
                   linewidth=4.5)
    second_ax.plot(alpha, error_sinus_real_vs_pade, color='purple', label='Error: real vs Pade sinus', linewidth=4.5)
    second_ax.set_yscale("log", base=10)
    second_ax.set_xlabel('$Alpha$')
    second_ax.set_ylabel('$Error$')
    second_ax.set_title('Taylor and Pade sinus errors (OY ax is on logarithmic scale)')
    second_ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_3_y_logarithm.png'))


def exercitiul_2(output_dir):
    start = 0.0
    stop = 2 * np.pi
    step = 0.05
    t = np.arange(start=start, stop=stop, step=step)

    amplitude = 1.0
    frequency = 2
    phases = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    phases_latex = ['\pi/6', '\pi/4', '\pi/3', '\pi/2']

    function_def = lambda A, freq, t, phase: A * np.sin(2 * np.pi * frequency * t + phase)

    results = [{'series_without_noise': function_def(amplitude, frequency, t, phase), 'phase': phase,
                'phase_latex': phase_latex}
               for phase, phase_latex in zip(phases, phases_latex)]

    x_without_noise_series = [result['series_without_noise'] for result in results]
    noise_series = [np.random.normal(loc=0.0, scale=1.0, size=t.shape[0]) for _ in range(len(x_without_noise_series))]
    snr_values = [0.1, 1, 10, 100]
    gammma_values = [np.sqrt((norm(x) ** 2) / (norm(z) ** 2) / snr) for x, z, snr in
                     zip(x_without_noise_series, noise_series, snr_values)]
    x_with_noise_series = [x + gamma * z for x, gamma, z in zip(x_without_noise_series, gammma_values, noise_series)]
    for result, x_with_noise in zip(results, x_with_noise_series):
        result['series_with_noise'] = x_with_noise

    fig, axes = plt.subplots(nrows=len(results), ncols=1, figsize=(10, 10))
    for result, ax in zip(results, axes):
        ax.plot(t, result['series_without_noise'], label='Without noise', color='red')
        ax.plot(t, result['series_with_noise'], label='With noise', color='blue')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Value')
        ax.set_title('$\phi={}$'.format(result['phase_latex']))
        ax.legend(loc='right', bbox_to_anchor=(1.25, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_2_.png'))


def exercitiul_1(output_dir):
    min = 0.0
    max = 2 * np.pi
    no_points = 10 ** 2
    t = np.linspace(start=min, stop=max, num=no_points)

    amplitude = 1.0
    phase = 0
    frequency = 2

    sin_simulation = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    cos_simulation = amplitude * np.cos(2 * np.pi * frequency * t + phase - np.pi / 2)
    results = [{'series': sin_simulation, 'plot_title': '$sin(t)$'},
               {'series': cos_simulation, 'plot_title': '$cos(t)$'}]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    for result, ax in zip(results, axes):
        ax.plot(t, result['series'])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Value')
        ax.set_title(result['plot_title'])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_1.png'))


if __name__ == '__main__':
    main()
