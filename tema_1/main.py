import math
import os
import scipy
import skimage
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from dahuffman import HuffmanCodec


def main():
    output_dir = './outputs'

    quantization_matrix = generate_quantization_matrix()

    # exercitiul 1 din tema

    gray_scale_image = scipy.datasets.ascent()
    compressed_gray_scale_image = compress_gray_scale_image_using_jpeg_format(gray_scale_image,
                                                                              quantization_matrix,
                                                                              os.path.join(output_dir, 'exercise_1'))
    decompressed_gray_scale_image = decompress_jpeg_gray_scale_image(compressed_gray_scale_image)
    plot_original_vs_decompressed_image(gray_scale_image, decompressed_gray_scale_image,
                                        os.path.join(output_dir, 'exercise_1'))

    # exercitiul 2 din tema
    rgb_image = scipy.datasets.face()
    compressed_rgb_image = compress_rgb_image_using_jpeg_format(rgb_image, quantization_matrix,
                                                                os.path.join(output_dir, 'exercise_2'))
    decompressed_rgb_image = decompress_rgb_image_using_jpeg_format(compressed_rgb_image)
    plot_original_vs_decompressed_image(rgb_image, decompressed_rgb_image, os.path.join(output_dir, 'exercise_2'),
                                        grayscale=False)


def decompress_rgb_image_using_jpeg_format(compressed_image):
    # validarea input-urilor
    if not isinstance(compressed_image, dict):
        raise ValueError('parameter compressed_image should be a dictionary!')

    required_keys = ['huffman_code_table', 'huffman_encoded_data', 'image_shape', 'quantization_matrix']
    available_keys = list(compressed_image.keys())
    if not all(key in required_keys for key in available_keys):
        raise ValueError(
            'the dictionary compressed_images should contain the following keys: {}\n'.format(available_keys))

    # extragerea informatiilor din dictionarul compressed_image
    quantization_matrix = compressed_image.get('quantization_matrix')
    huffman_code_table = compressed_image.get('huffman_code_table')
    huffman_encoded_data = compressed_image.get('huffman_encoded_data')
    image_shape = compressed_image.get('image_shape')

    # decodarea utilizand algoritmul lui Huffman
    decoded_data = HuffmanCodec(huffman_code_table).decode(huffman_encoded_data)

    # impartirea decoded_data pe fiecare canal de imagine
    no_channels = image_shape[-1]
    no_elements_decoded_data = len(decoded_data)
    no_elements_decoded_data_on_channel = no_elements_decoded_data // no_channels
    no_elements_decoded_data_on_channel_float = no_elements_decoded_data / no_channels

    if not no_elements_decoded_data_on_channel == no_elements_decoded_data_on_channel_float:
        raise ValueError('the number of encoded elements that not match the image number of channels!')

    decoded_data_on_channel = []
    for index in range(0, no_elements_decoded_data, no_elements_decoded_data_on_channel):
        decoded_data_on_channel.append(decoded_data[index:index + no_elements_decoded_data_on_channel])

    # aplicam algoritmul de decompresie JPEG pe fiecare channel
    ycbcr_image = np.zeros(image_shape)
    for channel_number, decoded_data in enumerate(decoded_data_on_channel):
        ycbcr_image[:, :, channel_number] = reverse_jpeg_algorithm(decoded_data, quantization_matrix, image_shape[:-1])

    # conversia imaginii din formatul YCBCR in formatul RGB
    rgb_image = skimage.color.ycbcr2rgb(ycbcr_image)

    return rgb_image


def compress_rgb_image_using_jpeg_format(rgb_image, quantization_matrix, output_dir):
    # validarea input-urilor
    if not (type(rgb_image) is np.ndarray and rgb_image.ndim == 3):
        raise ValueError('parameter image must be a 2D numpy array!')
    if not (rgb_image.dtype == np.uint8 and np.min(rgb_image) == 0 and np.max(rgb_image) == 255):
        raise ValueError('parameter image must be a 2D numpy array of integers with values between 0 and 255!')
    if not (type(quantization_matrix) is np.ndarray and quantization_matrix.ndim == 2):
        raise ValueError('parameter quantization_matrix must be a 2D numpy array!')

    # conversia imaginii din RGB YCBCR
    ycbcr_image = skimage.color.rgb2ycbcr(rgb_image)

    # obtinerea vectorilor zig-zag qunatizati pentru fiecare bloc si pentru fiecare channel
    no_channels = ycbcr_image.shape[2]

    zig_zag_vectors_on_channels = []
    for channel_number in range(no_channels):
        image = ycbcr_image[:, :, channel_number]
        zig_zag_vectors_img_on_blocks = compress_gray_scale_image_using_jpeg_format(image, quantization_matrix, None,
                                                                                    True)
        zig_zag_vectors_on_channels.extend(zig_zag_vectors_img_on_blocks)

    # aplicam encoding-ul conform algoritmului lui Huffman
    huffman_encoder = HuffmanCodec.from_data(zig_zag_vectors_on_channels)
    encoded_data = huffman_encoder.encode(zig_zag_vectors_on_channels)
    code_table = huffman_encoder.get_code_table()

    compressed_image = {'huffman_code_table': code_table,
                        'huffman_encoded_data': encoded_data,
                        'image_shape': rgb_image.shape,
                        'quantization_matrix': quantization_matrix}

    # salvarea datelor
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, 'compressed_image.pickle'), 'wb')
    pickle.dump(compressed_image, file)
    file.close()

    return compressed_image


def calculate_average_pixel():
    min_pixel = 0
    max_pixel = 255
    avg_pixel = np.mean([min_pixel, max_pixel])

    return np.round(avg_pixel).astype(int)


def plot_original_vs_decompressed_image(original_image, decompressed_image, output_dir, grayscale=True):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data = {'Original image': original_image,
            'Decompressed image': decompressed_image}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    for ax, (plot_title, image) in zip(axes, data.items()):
        if grayscale:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        ax.set_title(plot_title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'original_vs_decompressed_image.png'))


def reverse_jpeg_algorithm(decoded_data, quantization_matrix, image_shape):
    # salvarea vectorilor zig zag in liste separate
    block_shape = quantization_matrix.shape
    no_elemenets_zig_zag_vector = math.prod(block_shape)

    no_elements_decoded_data = len(decoded_data)
    no_zig_zag_vectors = no_elements_decoded_data // no_elemenets_zig_zag_vector
    no_zig_zag_vectors_float = no_elements_decoded_data / no_elemenets_zig_zag_vector
    if not (no_zig_zag_vectors_float == no_zig_zag_vectors):
        raise ValueError('the number of decoded data does not match the block shape!')

    zig_zag_vectors_quantization_image_blocks = []
    for index in range(0, no_elements_decoded_data, no_elemenets_zig_zag_vector):
        zig_zag_vector = decoded_data[index:index + no_elemenets_zig_zag_vector]
        zig_zag_vectors_quantization_image_blocks.append(zig_zag_vector)

    # transformarea vectorilor zig-zag in matricile corespunzatoare (matrici cuantizate)
    quantization_image_blocks = [generate_matrix_of_zig_zag_vector(zig_zag_vector, block_shape) for zig_zag_vector in
                                 zig_zag_vectors_quantization_image_blocks]

    # aplicam decuantizarea
    dequantized_image_blocks = [quantization_block * quantization_matrix for quantization_block in
                                quantization_image_blocks]

    # aplicam inversa transformatei cosinus discrete in 2D
    image_blocks = [scipy.fft.idctn(dequantized_image_block) for dequantized_image_block in dequantized_image_blocks]

    # adaugam pe fiecare bloc de imagine media pixelilor (ce a fost scazuta in cadrul algoritmului de encoding)
    avg_pixel = calculate_average_pixel()
    # image_blocks = [image_block + avg_pixel for image_block in image_blocks]

    # reconstruim imaginea din blocuri
    image = generate_image_from_blocks(image_blocks, image_shape, block_shape)

    return image


def decompress_jpeg_gray_scale_image(compressed_image):
    # validarea input-urilor
    if not isinstance(compressed_image, dict):
        raise ValueError('parameter compressed_image should be a dictionary!')

    required_keys = ['huffman_code_table', 'huffman_encoded_data', 'image_shape', 'quantization_matrix']
    available_keys = list(compressed_image.keys())
    if not all(key in required_keys for key in available_keys):
        raise ValueError(
            'the dictionary compressed_images should contain the following keys: {}\n'.format(available_keys))

    # extragerea informatiilor din dictionarul compressed_image
    quantization_matrix = compressed_image.get('quantization_matrix')
    huffman_code_table = compressed_image.get('huffman_code_table')
    huffman_encoded_data = compressed_image.get('huffman_encoded_data')
    image_shape = compressed_image.get('image_shape')

    # decode the encoded data using the Huffman algorithm
    decoded_data = HuffmanCodec(huffman_code_table).decode(huffman_encoded_data)

    # obtinerea imaginii
    image = reverse_jpeg_algorithm(decoded_data, quantization_matrix, image_shape)

    return image


def generate_image_from_blocks(blocks, image_shape, block_shape):
    image_no_rows, image_no_cols = image_shape
    block_no_rows, block_no_cols = block_shape

    image = np.zeros((image_no_rows, image_no_cols))
    index = 0
    for row_index in range(0, image_no_rows, block_no_rows):
        for col_index in range(0, image_no_cols, block_no_cols):
            image[row_index:row_index + block_no_rows, col_index:col_index + block_no_cols] = blocks[index]
            index += 1

    return image


def generate_matrix_of_zig_zag_vector(zig_zag_vector, block_shape):
    no_rows, no_cols = block_shape
    no_diagonals = no_rows + no_cols - 1

    index = 0
    matrix = np.zeros(block_shape)
    for diagonal_number in range(no_diagonals):
        all_elements_indexes_on_diagonal = [(row_index, col_index) for
                                            row_index in range(0, diagonal_number + 1) for col_index in
                                            range(0, diagonal_number + 1) if
                                            row_index + col_index == diagonal_number and row_index < no_rows and col_index < no_cols]

        if diagonal_number % 2 == 0:
            all_elements_indexes_on_diagonal = all_elements_indexes_on_diagonal[::-1]

        for row_index, col_index in all_elements_indexes_on_diagonal:
            matrix[row_index, col_index] = zig_zag_vector[index]
            index += 1

    return matrix


def compress_gray_scale_image_using_jpeg_format(image, quantization_matrix, output_dir, early_return=False):
    # validarea input-urilor
    if not (type(image) is np.ndarray and image.ndim == 2):
        raise ValueError('parameter image must be a 2D numpy array!')
    # if not (image.dtype == np.uint8 and np.min(image) == 0 and np.max(image) == 255):
    #    raise ValueError('parameter image must be a 2D numpy array of integers with values between 0 and 255!')
    if not (type(quantization_matrix) is np.ndarray and quantization_matrix.ndim == 2):
        raise ValueError('parameter quantization_matrix must be a 2D numpy array!')

    # impart imaginea gray scale in blocuri de 8x8
    image_blocks = generate_image_blocks(image)

    # scoatem media din semnal
    avg_pixel = calculate_average_pixel()
    # image_blocks = [image - avg_pixel for image in image_blocks]

    # aplicarea transformatei cosinus discreta (2D) pentru fiecare bloc
    dct_2d_image_blocks = [scipy.fft.dctn(block) for block in image_blocks]

    # cuantizarea in frecventa utilizand matricea de cuantizare
    quantization_image_blocks = [np.round(dct_2d_block / quantization_matrix).astype(int) for dct_2d_block in
                                 dct_2d_image_blocks]

    # vectorizarea zig-zag a cuantizarii
    zig_zag_quantization_image_blocks = [generate_zig_zag_vector_of_matrix(quantization_block) for quantization_block in
                                         quantization_image_blocks]
    zig_zag_quantization_image_blocks = list(itertools.chain.from_iterable(zig_zag_quantization_image_blocks))

    if early_return:
        return zig_zag_quantization_image_blocks

    # compresia utilizand algoritmul lui Huffman
    huffmann_encoder = HuffmanCodec.from_data(zig_zag_quantization_image_blocks)
    encoded_data = huffmann_encoder.encode(zig_zag_quantization_image_blocks)
    code_table = huffmann_encoder.get_code_table()

    compressed_image = {'huffman_code_table': code_table,
                        'huffman_encoded_data': encoded_data,
                        'image_shape': image.shape,
                        'quantization_matrix': quantization_matrix}

    # salvarea rezultatelor
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, 'compressed_image.pickle'), 'wb')
    pickle.dump(compressed_image, file)
    file.close()

    return compressed_image


def generate_zig_zag_vector_of_matrix(matrix):
    no_rows, no_cols = matrix.shape
    no_diagonals = no_rows + no_cols - 1

    zig_zag_matrix_elements = [[] for _ in range(no_diagonals)]
    for diagonal_number, diagonal_elements in enumerate(zig_zag_matrix_elements):
        all_elements_indexes_on_diagonal = [(row_index, col_index) for
                                            row_index in range(0, diagonal_number + 1) for col_index in
                                            range(0, diagonal_number + 1) if
                                            row_index + col_index == diagonal_number and row_index < no_rows and col_index < no_cols]

        if diagonal_number % 2 == 0:
            all_elements_indexes_on_diagonal = all_elements_indexes_on_diagonal[::-1]

        for row_index, col_index in all_elements_indexes_on_diagonal:
            element = matrix[row_index, col_index]
            diagonal_elements.append(element)

    zig_zag_matrix_elements = [element for row_elements in zig_zag_matrix_elements for element in row_elements]

    return zig_zag_matrix_elements


def generate_image_blocks(image, block_size=(8, 8)):
    if not (type(block_size) is tuple and len(block_size) == 2):
        raise ValueError('parameter block_size must be tuple of two elements!')

    block_no_rows, block_no_cols = block_size
    if not (type(block_no_rows) is int and block_no_rows > 0 and type(block_no_cols) is int and block_no_cols > 0):
        raise ValueError('the components of the block_size should be strictly positive integers!')

    if not (type(image) is np.ndarray and image.ndim == 2):
        raise ValueError('the parameter image should be a 2D numpy array!')

    image_no_rows, image_no_cols = image.shape

    image_blocks = []
    for row_index in range(0, image_no_rows, block_no_rows):
        for col_index in range(0, image_no_cols, block_no_cols):
            block = image[row_index:row_index + block_no_rows, col_index:col_index + block_no_cols]
            image_blocks.append(block)

    return image_blocks


def generate_quantization_matrix():
    quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 28, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

    return quantization_matrix


if __name__ == '__main__':
    main()
