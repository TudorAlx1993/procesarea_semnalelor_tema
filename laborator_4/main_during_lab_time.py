import os
import numpy as np
import matplotlib.pyplot as plt
from time import process_time


def main():
    output_dir = 'outputs_during_lab_time'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exercise_1(output_dir)
    exercise_2(output_dir)


def generate_fourier_matrix(N):
    fourier_matrix = np.array(
        [[np.exp(-2 * np.pi * 1j * row_index * col_index / N) for col_index in range(N)] for row_index in range(N)])

    return fourier_matrix


def exercise_1(output_dir):
    amplitude = 1.0
    frequency = 10
    phase = 0.0
    start = 0.0
    stop = 1.0

    generate_sin_signal = lambda t: amplitude * np.sin(2 * np.pi * t * frequency + phase)

    start_power = 7
    end_power = 14
    no_samples_values = [2 ** power for power in range(start_power, end_power)]
    t_vectors = [np.linspace(start=start, stop=stop, num=no_samples) for no_samples in no_samples_values]
    no_samples_and_signals = [(t_vector.shape[0], generate_sin_signal(t_vector)) for t_vector in t_vectors]

    time_results = []
    for no_samples, signal in no_samples_and_signals:
        start_time = process_time()
        fourier_matrix = generate_fourier_matrix(N=no_samples)
        X_my_version = np.matmul(fourier_matrix, signal.reshape(-1, 1)).flatten()
        end_time = process_time()
        my_duration = end_time - start_time

        start_time = process_time()
        X_numpy_version = np.fft.fft(signal)
        end_time = process_time()
        numpy_duration = end_time - start_time

        assert np.allclose(X_my_version, X_numpy_version)

        time_results.append((no_samples, my_duration, numpy_duration))

    no_samples = [no_samples for (no_samples, _, _) in time_results]
    my_durations = [my_duration for (_, my_duration, _) in time_results]
    numpy_durations = [numpy_duration for (_, _, numpy_duration) in time_results]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.plot(no_samples, np.log10(my_durations), label='My code', color='blue', linewidth=4.5)
    ax.plot(no_samples, np.log10(numpy_durations), label='Numpy FFT', color='red', linewidth=4.5)
    ax.set_xlabel('No samples')
    ax.set_ylabel('Log of time (seconds)')
    ax.set_title('Execution time: my Fourier transform vs numpy implementation')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'exercise_1.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_1.pdf'), format='pdf')


def exercise_2(output_dir):
    pass


if __name__ == '__main__':
    main()
