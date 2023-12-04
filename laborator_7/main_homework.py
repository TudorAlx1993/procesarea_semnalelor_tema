import os
import scipy
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_homework'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ex_1(output_dir)
    ex_2(output_dir, 1.05)
    ex_3(output_dir)
    ex_4(output_dir)


def ex_4(output_dir):
    pass


def ex_2(output_dir, snr):
    original_image = scipy.misc.face(gray=True)
    X_original_image = np.fft.rfft2(original_image)

    no_freq_components = np.sum(X_original_image.shape)
    no_freq_components_to_delete = no_freq_components - int(no_freq_components / snr)

    X_original_image_abs_db = 20 * np.log10(np.abs(X_original_image))
    threshold = np.partition(X_original_image_abs_db.flatten(), -no_freq_components_to_delete)[
        -no_freq_components_to_delete]
    X_compressed_image = X_original_image.copy()
    X_compressed_image[X_original_image_abs_db >= threshold] = 0
    compressed_image = np.fft.irfft2(X_compressed_image)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    for ax, image, title in zip(axes.flatten(), [original_image, compressed_image],
                                ['Original image', 'Compressed image']):
        ax.imshow(image, cmap=plt.cm.gray)
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_2.png'))
    fig.savefig(os.path.join(output_dir, 'ex_2.pdf'), format='pdf')


def ex_3(output_dir):
    pass


def save_plot_of_image(image, file_name):
    fig, (left_ax, right_ax) = plt.subplots(nrows=1, ncols=2)
    left_ax.imshow(image)
    left_ax.set_title('Color image')
    right_ax.imshow(image, cmap=plt.cm.gray)
    right_ax.set_title('Grayscale image')
    for ax in [left_ax, right_ax]:
        ax.set_xlabel('Column index')
        ax.set_ylabel('Row index')
    fig.tight_layout()
    fig.savefig('{}.png'.format(file_name))
    fig.savefig('{}.pdf'.format(file_name), format='pdf')


def save_spectrogram_of_image(X_image, file_name):
    X_image_abs_db = 20 * np.log10(np.abs(X_image + 1e-10))

    fig, ax = plt.subplots()
    color_bar = ax.imshow(X_image_abs_db)
    fig.colorbar(color_bar, ax=ax)
    ax.set_title('Image spectrogram')
    fig.savefig('{}.png'.format(file_name))
    fig.savefig('{}.pdf'.format(file_name), format='pdf')


def ex_1(output_dir):
    # punctul a)
    no_rows, no_cols = 900, 900
    row_indexes, col_indexes = np.indices((no_rows, no_cols))
    image = np.sin(2 * np.pi * row_indexes + 3 * np.pi * col_indexes)
    save_plot_of_image(image, os.path.join(output_dir, 'ex_1_pct_a_image'))
    X_image = np.fft.fft2(image)
    save_spectrogram_of_image(X_image, os.path.join(output_dir, 'ex_1_pct_a_image_spectrogram'))

    # punctul b)
    no_rows, no_cols = 900, 900
    row_indexes, col_indexes = np.indices((no_rows, no_cols))
    image = np.sin(4 * np.pi * row_indexes) + np.cos(6 * np.pi * col_indexes)
    save_plot_of_image(image, os.path.join(output_dir, 'ex_1_pct_b_image'))
    X_image = np.fft.fft2(image)
    save_spectrogram_of_image(X_image, os.path.join(output_dir, 'ex_1_pct_b_image_spectrogram'))

    # punctul c)
    no_rows, ncols = 900, 900
    X_image = np.zeros(shape=(no_rows, no_cols))
    X_image[0][5] = X_image[0][ncols - 5] = 1
    image = np.fft.ifft2(X_image).real
    save_plot_of_image(image, os.path.join(output_dir, 'ex_1_pct_c_image'))
    save_spectrogram_of_image(X_image, os.path.join(output_dir, 'ex_1_pct_c_image_spectrogram'))

    # punctul d)
    no_rows, ncols = 900, 900
    X_image = np.zeros(shape=(no_rows, no_cols))
    X_image[5][0] = X_image[no_rows - 5][0] = 1
    image = np.fft.ifft2(X_image).real
    save_plot_of_image(image, os.path.join(output_dir, 'ex_1_pct_d_image'))
    save_spectrogram_of_image(X_image, os.path.join(output_dir, 'ex_1_pct_d_image_spectrogram'))

    # punctul e)
    no_rows, ncols = 900, 900
    X_image = np.zeros(shape=(no_rows, no_cols))
    X_image[5][5] = X_image[no_rows - 5][ncols - 5] = 1
    image = np.fft.ifft2(X_image).real
    save_plot_of_image(image, os.path.join(output_dir, 'ex_1_pct_e_image'))
    save_spectrogram_of_image(X_image, os.path.join(output_dir, 'ex_1_pct_e_image_spectrogram'))


if __name__ == '__main__':
    main()
