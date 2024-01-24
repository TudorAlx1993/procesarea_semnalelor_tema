import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    # exemplul 1: ascent image din scipy
    image = scipy.datasets.ascent().astype('int32')
    template_output_dir = './outputs/example_1'

    # pasul 1: gaussian blurring
    output_dir = os.path.join(template_output_dir, 'gaussian_blurring')
    kernel_size = (5, 5)
    sigma = 1.0
    blurred_image_with_convolution = gaussian_blur_with_convolution(image, kernel_size, sigma, output_dir)
    blurred_image_in_frequency = gaussian_blur_in_frequency(image, 100.0, output_dir)
    blurred_image_with_opencv = gaussian_blur_with_opencv(image.astype('uint8'), kernel_size, sigma, output_dir)
    # ne uitam la diferentele intre cele 3 variante de gaussian blurring
    data = {'original image': image,
            'blurring with convolution\nkernel_size={} sigma={}'.format(kernel_size,
                                                                        sigma): blurred_image_with_convolution,
            'blurring in frequency': blurred_image_in_frequency,
            'blurring with opencv\nkernel_size={} sigma={}'.format(kernel_size, sigma): blurred_image_with_opencv}
    plot_images(data, 'gaussian_blurring.png', output_dir)

    # pasul 2: calculul gradientului
    output_dir = os.path.join(template_output_dir, 'gradients')
    sobel_gradient_magnitude, sobel_gradient_orientation = compute_gradients(blurred_image_with_convolution, 'sobel',
                                                                             output_dir)
    prewitt_gradient_magnitude, prewitt_gradient_orientation = compute_gradients(blurred_image_with_convolution,
                                                                                 'prewitt',
                                                                                 output_dir)
    scharr_gradient_magnitude, scharr_gradient_orientation = compute_gradients(blurred_image_with_convolution,
                                                                               'scharr',
                                                                               output_dir)


def compute_gradients(image, method, output_dir=None):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')

    if method == 'sobel':
        h_x, h_y = sobel_operators()
    elif method == 'prewitt':
        h_x, h_y = prewitt_operators()
    elif method == 'scharr':
        h_x, h_y = scharr_operators()
    else:
        raise ValueError('algorithm not implemented for method={}!'.format(method))

    G_x = scipy.ndimage.convolve(image, h_x)
    G_y = scipy.ndimage.convolve(image, h_y)
    gradient_magnitude = np.sqrt(G_x ** 2 + G_y ** 2)
    gradient_magnitude *= 255.0 / np.max(gradient_magnitude)
    gradient_orientation = np.arctan2(G_y, G_x)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data = {'blurred image': image,
                'horizontal gradient': G_x,
                'vertical gradient': G_y,
                'gradient magnitude': gradient_magnitude,
                'gradient orientation': gradient_orientation}
        fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 10))
        for ax, (plot_title, img) in zip(axes, data.items()):
            ax.imshow(img, cmap='gray')
            ax.set_title(plot_title.capitalize())
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '{}_gradients.png'.format(method)))

    return gradient_magnitude, gradient_orientation


def sobel_operators():
    h_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    h_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

    return h_x, h_y


def scharr_operators():
    h_x = np.array([[3, 0, -3],
                    [10, 0, -10],
                    [3, 0, -3]])
    h_y = np.array([[3, 10, 3],
                    [0, 0, 0],
                    [-3, -10, -3]])

    return h_x, h_y


def prewitt_operators():
    h_x = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])
    h_y = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])

    return h_x, h_y


def plot_images(data, file_name, output_dir, plot_grayscale=True):
    if not isinstance(data, dict):
        raise ValueError('parameter data should be a dictionary!')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 10))
    for ax, (plot_title, img) in zip(axes, data.items()):
        ax.imshow(img, cmap='gray') if plot_grayscale else ax.imshow(img)
        ax.set_title(plot_title.capitalize())
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, file_name))


def gaussian_blur_with_opencv(image, kernel_size, sigma, output_dir):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')
    # aici putem avea si sigma=0 si lasam opencv sa determine valoarea lui in functie de kernel_size
    if not (isinstance(sigma, float) and sigma >= 0):
        raise ValueError('parameter sigma should be a strictly positive number!')
    if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
        raise ValueError('parameter kernel_size should be a tuple with two elements!')
    no_rows, no_cols = kernel_size
    if not (isinstance(no_rows, int) and isinstance(no_cols, int) and no_rows % 2 == 1 and no_cols % 2 == 1):
        raise ValueError('the dimensions of the kernel should be odd integers!')

    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data = {'original image': image,
                'blurred image: kernel_size={} sigma={}'.format(kernel_size, sigma): blurred_image}
        fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 10))
        for ax, (plot_title, img) in zip(axes, data.items()):
            ax.imshow(img, cmap='gray')
            ax.set_title(plot_title.capitalize())
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'gaussian_blurring_with_opencv.png'))

    return blurred_image


def gaussian_blur_in_frequency(image, sigma, output_dir):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')
    if not (isinstance(sigma, float) and sigma > 0):
        raise ValueError('parameter sigma should be a strictly positive number!')

    image_spectrum = np.fft.fft2(image)
    image_centered_spectrum = np.fft.fftshift(image_spectrum)

    no_rows, no_cols = image.shape
    kernel_center = np.array([no_rows // 2, no_cols // 2])
    gaussian_kernel = np.array([[np.exp(
        -squared_distance_between_points(np.array([row_index, col_index]), kernel_center) / (2 * sigma ** 2)) for
        col_index in range(no_cols)] for row_index in range(no_rows)])
    normalization_constant = 1.0 / (2 * np.pi * sigma ** 2)
    gaussian_kernel *= normalization_constant

    filtered_image_centered_spectrum = image_centered_spectrum * gaussian_kernel
    blurred_image = np.fft.ifft2(np.fft.ifftshift(filtered_image_centered_spectrum)).real

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data = {'original image': image,
                'spectrum of image': 20 * np.log10(np.abs(image_spectrum)),
                'centered spectrum of image': 20 * np.log10(np.abs(image_centered_spectrum)),
                'gaussian kernel': gaussian_kernel,
                'filtered centered spectrum of image': 20 * np.log10(np.abs(filtered_image_centered_spectrum)),
                'blurred image': blurred_image}
        fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 10))
        for ax, (plot_title, img) in zip(axes, data.items()):
            ax.imshow(img, cmap='gray')
            ax.set_title(plot_title.capitalize())
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'gaussian_blurring_in_frequency.png'))

    return blurred_image


def squared_distance_between_points(first_point, second_point):
    if not (isinstance(first_point, np.ndarray) and first_point.ndim == 1 and first_point.size == 2):
        raise ValueError('parameter first_point should be a 1D numpy array with exact two elements!')
    if not (isinstance(second_point, np.ndarray) and second_point.ndim == 1 and second_point.size == 2):
        raise ValueError('parameter second_point should be a 1D numpy array with exact two elements!')

    squared_distance = np.sum((first_point - second_point) ** 2)

    return squared_distance


def gaussian_blur_with_convolution(image, kernel_size, sigma, output_dir):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')
    if not (isinstance(sigma, float) and sigma > 0):
        raise ValueError('parameter sigma should be a strictly positive number!')
    if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
        raise ValueError('parameter kernel_size should be a tuple with two elements!')
    no_rows, no_cols = kernel_size
    if not (isinstance(no_rows, int) and isinstance(no_cols, int) and no_rows % 2 == 1 and no_cols % 2 == 1):
        raise ValueError('the dimensions of the kernel should be odd integers!')

    # aici practic iau in calcul distanta fata de centrul kernelului
    kernel_center = np.array([no_rows // 2, no_cols // 2])
    gaussian_kernel = np.array([[np.exp(
        -squared_distance_between_points(np.array([row_index, col_index]), kernel_center) / (2 * sigma ** 2)) for
        col_index in range(no_cols)] for row_index in range(no_rows)])
    normalization_constant = 1.0 / (2 * np.pi * sigma ** 2)
    gaussian_kernel *= normalization_constant

    blurred_image = scipy.ndimage.convolve(image, gaussian_kernel)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data = {'original image': image,
                'gaussian kernel: size={} sigma={}'.format(kernel_size, sigma): gaussian_kernel,
                'blurred image': blurred_image}
        fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 10))
        for ax, (plot_title, img) in zip(axes, data.items()):
            ax.imshow(img, cmap='gray')
            ax.set_title(plot_title.capitalize())
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'gaussian_blurring_with_convolution.png'))

    return blurred_image


if __name__ == '__main__':
    main()
