import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


def main():
    output_dir = './outputs_in_timpul_laboratorului'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exercitiul_1(output_dir)
    exercitiul_2(output_dir)
    exercitiul_8(output_dir)


def exercitiul_8(output_dir):
    min,max=-np.pi/2,np.pi/2
    no_points=10**5
    alpha=np.linspace(min,max,num=no_points)


def exercitiul_2(output_dir):
    min = 0.0
    max = np.round(2 * np.pi,0)+1
    step_cont=0.05
    t = np.arange(start=min, stop=max+step_cont, step=step_cont)

    amplitude = 1.0
    frequency = 2
    phases = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    phases_latex = ['\pi/6', '\pi/4', '\pi/3', '\pi/2']

    function_def = lambda A, freq, t, phase: A * np.sin(2 * np.pi * frequency * t + phase)

    results_cont = [{'series': function_def(amplitude, frequency, t, phase), 'phase': phase, 'phase_latex': phase_latex}
                    for phase, phase_latex in zip(phases, phases_latex)]

    step_discrete = 1 / frequency
    n = np.arange(start=min, stop=max+step_discrete, step=step_discrete)

    results_discrete_wo_noise = [{'series': function_def(amplitude, frequency, n, result_cont['phase'])} for
                                 result_cont in results_cont]

    z_values = [np.random.normal(loc=0.0, scale=1.0, size=n.shape[0]) for _ in range(len(results_cont))]
    snr_values = [0.1, 1, 10, 100]
    x_values = [value['series'] for value in results_discrete_wo_noise]
    gammma_values = [np.sqrt(norm(x)**2 / norm(z)**2 / snr) for x, z, snr in
                     zip(x_values, z_values, snr_values)]

    results_discrete_noise = [{'series': x + gamma * z} for x, gamma, z in zip(x_values, gammma_values, z_values)]

    fig, axes = plt.subplots(nrows=len(results_cont), ncols=1, figsize=(10, 10))
    for result_cont,result_discrete_noise, ax in zip(results_cont,results_discrete_noise, axes):
        ax.plot(t, result_cont['series'],label='$x(t)$')
        ax.stem(n,result_discrete_noise['series'],label='$x[n]+\gamma[n]$')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Value')
        ax.set_title('$\phi={}$'.format(result_cont['phase_latex']))
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_2_plot_1.png'))


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
