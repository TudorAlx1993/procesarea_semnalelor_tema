import os
import scipy
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_during_lab_time'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ex_1(100, output_dir)


def compute_1d_convolution(x, y):
    if not (type(x) == type(y) and type(x) is np.ndarray):
        raise ValueError('x and y parameters should be numpy arrays!')

    if not (x.ndim == 1 and y.ndim == 1):
        raise ValueError('x and y paramters should be 1D numpy arrays!')

    x_no_elements = x.shape[0]
    y_no_elements = y.shape[0]
    convolution_no_elements = x_no_elements + y_no_elements - 1
    convolution = np.zeros(convolution_no_elements)
    for first_index in range(convolution_no_elements):
        print(first_index)
        convolution[first_index] = np.array(
            [y[second_index] * x[first_index - second_index] for second_index in range(first_index)]).sum()

    return convolution


def ex_1(n, output_dir):
    # simulez din distributia uniforma pe intervalul [0,1)
    x = np.random.rand(n)

    no_iterations = 3
    results = [('original signal (random from uniform distribution over the [0,1) interval)', x)]
    for iteration in range(1, no_iterations + 1):
        # x=compute_1d_convolution(x,x)
        x = np.convolve(x, x)
        # pentru scalare: axa OY in graficele de mai jos sa fie intre 0 si 1
        # scalarea nu e obligatorie, e doar sa nu imi afiseze valori mari pe axa OY
        x = x / np.max(x)
        results.append(('iteration {}'.format(iteration), x))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axes = axes.flatten()
    for (info, convolution), ax in zip(results, axes):
        ax.plot(np.arange(1, 1 + convolution.shape[0]), convolution, color='steelblue', linewidth=5.0)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.set_title(info.capitalize())
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_1.png'))
    fig.savefig(os.path.join(output_dir, 'ex_1.pdf'), format='pdf')


if __name__ == '__main__':
    main()
