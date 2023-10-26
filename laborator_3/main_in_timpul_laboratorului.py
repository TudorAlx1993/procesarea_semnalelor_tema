import os

import numpy
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_in_timpul_laboratorului'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    N = 8
    # exercitiul_1(N, output_dir)
    exercitiul_2(output_dir)


def get_fourier_matrix(N, abs_error=10 ** -10):
    fourier_matrix = np.array(
        [[np.exp(2 * np.pi * 1j * row_index * col_index / N) for col_index in range(N)] for row_index in range(N)])

    assert np.linalg.norm(
        np.abs(np.matmul(fourier_matrix.transpose().conjugate(), fourier_matrix) - N * np.identity(N)),
        ord='fro') <= abs_error

    return fourier_matrix


def exercitiul_2(output_dir):
    amplitude = 1.0
    phase = 0.0
    frequency = 2.0

    start, stop = 0.0, 1.0
    no_samples = 10 ** 3
    t = np.linspace(start, stop, no_samples)
    x = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    start, stop = 0.0, 1.0
    n = np.linspace(start, stop, no_samples)
    y = np.array([x[index] * np.exp(-2 * np.pi * 1j * step) for index, step in zip(range(no_samples), n)])

    fig, (left_ax, right_ax) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    left_ax.plot(t, x, color='blue', linewidth=4.5)
    left_ax.set_xlabel('Time')
    left_ax.set_ylabel('Value')
    left_ax.set_title('Sinosoidal signal with $\omega$={:.2f} Hertz'.format(frequency))
    right_ax.plot(y.real, y.imag, color='red')
    right_ax.set_xlabel('Real part')
    right_ax.set_ylabel('Imaginary part')
    right_ax.set_title('Complex unitary circle')
    fig.savefig(os.path.join(output_dir, 'problema_2.png'))


def exercitiul_1(N, output_dir, abs_error=10 ** -10):
    fourier_matrix = get_fourier_matrix(N, abs_error)

    file = open(os.path.join(output_dir, 'problema_1_fourier_matrix.npy'), 'wb')
    np.save(file, fourier_matrix)
    file.close()

    fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(10, 15))
    for ax, row_index in zip(axes, range(fourier_matrix.shape[0])):
        ax.plot(fourier_matrix[row_index].real, color='blue', label='Real part', linewidth=4.5)
        ax.plot(fourier_matrix[row_index].imag, color='red', label='Imag part', linewidth=4.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Matrix row={}'.format(row_index))
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'problema_1.png'))

    return fourier_matrix


if __name__ == '__main__':
    main()
