import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_homework'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    exercise_0()
    exercise_1(N=8, output_dir=output_dir)
    exercise_2(output_dir=output_dir)
    exercise_3(output_dir=output_dir)


def exercise_0(abs_error=10 ** -10):
    # calculate fourier transformation using 4 different methods

    amplitude = 1.0
    frequency = 4
    phase = 0.0
    start = 0.0
    stop = 1.0
    step = 0.005
    t = np.arange(start=start, stop=stop, step=step)
    x = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    vector_dim = x.shape[0]

    X_v1 = np.zeros(shape=x.shape, dtype=np.complex_)
    for first_index in range(vector_dim):
        sum = 0
        for second_index in range(vector_dim):
            sum += x[second_index] * np.exp(2 * np.pi * 1j * second_index * first_index / vector_dim)
        X_v1[first_index] = sum

    X_v2 = np.zeros_like(X_v1)
    for first_index in range(vector_dim):
        X_v2[first_index] = np.sum(x * np.array(
            [np.exp(2 * np.pi * 1j * second_index * first_index / vector_dim) for second_index in range(vector_dim)]))

    fourier_matrix = generate_fourier_matrix(N=vector_dim)
    X_v3 = np.matmul(fourier_matrix, x.reshape(-1, 1)).flatten()

    assert np.allclose(X_v1, X_v2)
    assert np.allclose(X_v1, X_v3)
    assert np.allclose(X_v2, X_v3)


def generate_fourier_matrix(N):
    fourier_matrix = np.array(
        [[np.exp(2 * np.pi * 1j * row_index * col_index / N) for col_index in range(N)] for row_index in range(N)])

    return fourier_matrix


def exercise_1(N, output_dir, abs_error=10 ** -10):
    fourier_matrix = generate_fourier_matrix(N)

    assert np.linalg.norm(
        np.abs(np.matmul(fourier_matrix.transpose().conjugate(), fourier_matrix) - N * np.identity(N)),
        ord='fro') <= abs_error

    # or using np.allclose
    assert np.allclose(
        np.linalg.norm(np.abs(np.matmul(fourier_matrix.transpose().conjugate(), fourier_matrix) - N * np.identity(N)),
                       ord='fro'), abs_error)

    file = open(os.path.join(output_dir, 'exercise_1_fourier_matrix.npy'), 'wb')
    np.save(file, fourier_matrix)
    file.close()

    fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(10, 15))
    for ax, row_index in zip(axes, range(fourier_matrix.shape[0])):
        ax.plot(fourier_matrix[row_index].real, color='blue', label='Real part', linewidth=4.5)
        ax.plot(fourier_matrix[row_index].imag, color='red', label='Imaginary part', linewidth=4.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Matrix row={}'.format(row_index))
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_1_matrix_rows_plot.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_1_matrix_rows_plot.pdf'), format='pdf')

    return fourier_matrix


def exercise_2(output_dir):
    pass


def exercise_3(output_dir):
    amplitudes = [1.0, 1.5, 2.0]
    frequencies = [10, 25, 50]
    phases = [0, np.pi / 6, np.pi / 4]

    start = 0.0
    stop = 1.0
    step = 0.005
    time = np.arange(start=start, stop=stop, step=step)

    individual_signals = np.array(
        [amplitude * np.sin(2 * np.pi * frequency * time + phase) for amplitude, frequency, phase in
         zip(amplitudes, frequencies, phases)])
    signal = np.sum(individual_signals, axis=0)
    vector_dim = signal.shape[0]

    omega_values = np.arange(0, max(frequencies) * 2)
    abs_fourier_transformation = [np.abs(
        np.sum([signal[index] * np.exp(2 * np.pi * 1j * index * omega / vector_dim) for index in range(vector_dim)]))
        for omega in omega_values]

    for frequency in frequencies:
        assert not np.allclose(0.0, abs_fourier_transformation[frequency])

    for omega in omega_values:
        if omega not in frequencies:
            assert np.allclose(0.0, abs_fourier_transformation[omega])

    fig, (left_ax, right_ax) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    left_ax.plot(time, signal, color='red', linewidth=3.0)
    left_ax.set_xlabel('Time (s)')
    left_ax.set_ylabel('Value')
    left_ax.set_title('Signal evolution')
    right_ax.stem(omega_values, abs_fourier_transformation)
    right_ax.set_xlabel('Frequency (Hz)')
    right_ax.set_ylabel('|X($\omega$)|')
    right_ax.set_title('Signal dominant frequencies')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_3.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_3.pdf'), format='pdf')


if __name__ == '__main__':
    main()
