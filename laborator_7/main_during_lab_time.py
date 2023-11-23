import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_during_lab_time'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ex_1(output_dir)


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


def save_specter_of_image(X_image, file_name):
    X_imag_abs_db = 20 * np.log10(np.abs(X_image))

    fig, ax = plt.subplots()
    ax.imshow(X_imag_abs_db)
    ax.set_title('Image specter')
    fig.savefig('{}.png'.format(file_name))
    fig.savefig('{}.pdf'.format(file_name), format='pdf')


def ex_1(output_dir):
    # punctul a)
    no_rows, no_cols = 600, 900
    image = np.array(
        [[np.sin(2 * np.pi * row_index + 3 * np.pi * col_index) for col_index in range(no_cols)] for row_index in
         range(no_rows)])
    save_plot_of_image(image, os.path.join(output_dir, 'ex_1_pct_a_image'))
    X_image = np.fft.fft2(image)
    save_specter_of_image(X_image, os.path.join(output_dir, 'ex_1_pct_a_specter_of_image'))

    # punctul b)
    no_rows, no_cols = 600, 900
    image = np.array(
        [[np.sin(4 * np.pi * row_index) + np.cos(6 * np.pi * col_index) for col_index in range(no_cols)] for row_index
         in
         range(no_rows)])
    save_plot_of_image(image, os.path.join(output_dir, 'ex_1_pct_b_image'))
    X_image = np.fft.fft2(image) + 10 ** -8
    save_specter_of_image(X_image, os.path.join(output_dir, 'ex_1_pct_b_specter_of_image'))


if __name__ == '__main__':
    main()
