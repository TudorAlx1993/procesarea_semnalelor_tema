import math
import os
import scipy
import pickle
import itertools
import skimage
import numpy as np
import matplotlib.pyplot as plt
from dahuffman import HuffmanCodec
from PIL import Image
import cv2


def main():
    output_dir = './outputs'

    quantization_matrix = generate_quantization_matrix()

    # exercitiul 1 din tema
    gray_scale_image = scipy.datasets.ascent()
    compressed_gray_scale_image = compress_gray_scale_image_using_jpeg_format(gray_scale_image,
                                                                              quantization_matrix,
                                                                              os.path.join(output_dir, 'exercise_1'))
    decompress_gray_scale_image = decompress_gray_scale_image_using_jpeg_format(compressed_gray_scale_image)
    plot_original_vs_decompressed_image(gray_scale_image, decompress_gray_scale_image,
                                        os.path.join(output_dir, 'exercise_1'))

    # exercitiul 2 din tema
    rgb_image = scipy.datasets.face()
    compressed_rgb_image = compress_rgb_image_using_jpeg_algorithm(rgb_image, quantization_matrix,
                                                                   os.path.join(output_dir, 'exercise_2'))
    decompressed_rgb_image = decompress_rgb_image_using_jpeg_algorithm(compressed_rgb_image)
    plot_original_vs_decompressed_image(rgb_image, decompressed_rgb_image, os.path.join(output_dir, 'exercise_2'))

    # exercitiul 3 din tema
    target_mse = 0.7
    compressed_gray_scale_image = compress_gray_scale_image_using_jpeg_format_given_mse(gray_scale_image,
                                                                                        quantization_matrix,
                                                                                        target_mse,
                                                                                        os.path.join(
                                                                                            output_dir,
                                                                                            'exercise_3'))
    decompress_gray_scale_image = decompress_gray_scale_image_using_jpeg_format(compressed_gray_scale_image)
    achived_mse = np.mean((gray_scale_image - decompress_gray_scale_image) ** 2)
    file_content = 'Target MSE: {:.4f}\nAchived MSE: {:.4f}'.format(target_mse, achived_mse)
    write_to_file(file_content, os.path.join(output_dir, 'exercise_3/achived_vs_target_mse.txt'))
    plot_original_vs_decompressed_image(gray_scale_image, decompress_gray_scale_image,
                                        os.path.join(output_dir, 'exercise_3'))

    # exercitiul 4 din tema
    video = read_video_file('./inputs/sample_video.mp4', (512, 512))
    # video = video[:10, :, :, :]
    compressed_video = compress_video_using_jpeg_format(video, quantization_matrix,
                                                        os.path.join(output_dir, 'exercise_4'))
    decompressed_video = decompress_video_using_jpeg_format(compressed_video)
    save_video(decompressed_video, os.path.join(output_dir, 'exercise_4/decompressed_video.mp4'))


def write_to_file(file_content, file_name):
    file = open(file_name, 'w')
    file.write(file_content)
    file.close()


def compress_gray_scale_image_using_jpeg_format_given_mse(image, quantization_matrix, mse_target, output_dir):
    # validarea input-urilor
    if not (type(image) is np.ndarray and image.ndim == 2):
        raise ValueError('parameter image must be a 2D numpy array!')
    if not (image.dtype == np.uint8 and np.min(image) == 0 and np.max(image) == 255):
        raise ValueError('parameter image must be a 2D numpy array of integers with values between 0 and 255!')
    if not (type(quantization_matrix) is np.ndarray and quantization_matrix.ndim == 2):
        raise ValueError('parameter quantization_matrix must be a 2D numpy array!')
    if not (type(mse_target) is float and mse_target > 0.0):
        raise ValueError('parameter mse_target should be a strictly positive float number!')

    def encode_to_jpeg(image, quantization_matrix, correction_factor):
        image_blocks = np.array(generate_image_blocks(image))
        # image_blocks = np.array([image - calculate_average_pixel() for image in image_blocks])
        dct_2d_image_blocks = np.array([scipy.fft.dctn(block) for block in image_blocks])
        quant_image_blocks = np.round(
            np.divide(dct_2d_image_blocks, quantization_matrix * correction_factor)).astype(int)

        return quant_image_blocks

    def deconde_from_jpeg(quant_image_blocks, quantization_matrix, correction_factor):
        dequant_image_blocks = np.multiply(quant_image_blocks, quantization_matrix * correction_factor)
        decompressed_image_blocks = np.array([scipy.fft.idctn(block) for block in dequant_image_blocks])
        decompressed_image = generate_image_from_blocks(decompressed_image_blocks, image.shape,
                                                        quantization_matrix.shape)
        decompressed_image = np.array(Image.fromarray(np.uint8(decompressed_image)))

        return decompressed_image

    def generate_compression_mse(image, quantization_matrix, correction_factor):
        quant_image_blocks = encode_to_jpeg(image, quantization_matrix, correction_factor)
        decompressed_image = deconde_from_jpeg(quant_image_blocks, quantization_matrix, correction_factor)

        mse = np.mean((image - decompressed_image) ** 2)

        return mse

    # MSE (mean squared error) depinde de matricea de quantizare Q
    # astfel putem influenta valoarea MSE in functie de magnitudinea lui Q
    # vom folosi Q'=Q*factor_corectie, unde factor_corectie este un strict pozitiv
    # cu cat factor_corectie este mai mare cu atat MSE este mai mare
    # cu cat factor_corectie este mai mic cu atat MSE este mai mic
    # incepem cu un factor de corectie mare

    correction_factor = 100

    mse_current = generate_compression_mse(image, quantization_matrix, correction_factor)
    while mse_current > mse_target:
        # ajustam factorul de corectie in functie de raportul intre mse_target si mse_current
        correction_factor = correction_factor * (mse_target / mse_current)
        mse_current = generate_compression_mse(image, quantization_matrix, correction_factor)

    achived_mse = generate_compression_mse(image, quantization_matrix, correction_factor)

    quant_image_blocks = encode_to_jpeg(image, quantization_matrix, correction_factor)
    zig_zag_vector_image_blocks = [generate_zig_zag_vector_of_matrix(matrix) for matrix in quant_image_blocks]
    zig_zag_vector_image_blocks = list(itertools.chain.from_iterable(zig_zag_vector_image_blocks))

    code_table, encoded_data = apply_huffman_encoding(zig_zag_vector_image_blocks)

    compressed_image = {'huffman_code_table': code_table,
                        'huffman_encoded_data': encoded_data,
                        'image_shape': image.shape,
                        'quantization_matrix': quantization_matrix * correction_factor}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, 'compressed_image.pickle'), 'wb')
    pickle.dump(compressed_image, file)
    file.close()

    return compressed_image


def save_video(video, file_name):
    no_frames, no_rows, no_cols, no_channels = video.shape

    codecs = cv2.VideoWriter_fourcc(*'mp4v')
    frame_rate = 30
    video_writer = cv2.VideoWriter(file_name, codecs, frame_rate, (no_rows, no_cols))

    for frame in video:
        frame = np.uint8(frame)
        video_writer.write(frame)

    video_writer.release()


def decompress_video_using_jpeg_format(compressed_video):
    # validarea input-urilor
    if not isinstance(compressed_video, dict):
        raise ValueError('parameter compressed_image should be a dictionary!')

    required_keys = ['huffman_code_table', 'huffman_encoded_data', 'video_shape', 'quantization_matrix']
    available_keys = list(compressed_video.keys())
    if not all(key in required_keys for key in available_keys):
        raise ValueError(
            'the dictionary compressed_images should contain the following keys: {}\n'.format(available_keys))

    # extragerea informatiilor din dictionarul compressed_image
    quantization_matrix = compressed_video.get('quantization_matrix')
    huffman_code_table = compressed_video.get('huffman_code_table')
    huffman_encoded_data = compressed_video.get('huffman_encoded_data')
    video_shape = compressed_video.get('video_shape')

    # decoding-ul utilizand algoritmul lui Huffman
    decoded_data = apply_huffman_decoding(huffman_code_table, huffman_encoded_data)

    # generam templat-ul video-ului
    video = np.zeros(video_shape)

    # reconstruim video-ul pe fiecare frame
    no_frames = video_shape[0]
    no_elements_decoded_data = len(decoded_data)

    if no_elements_decoded_data % no_frames != 0:
        raise ValueError('the number of zig zag vectors does not match the number of frames!')

    zig_zag_vectors_video = np.array(decoded_data).reshape(no_frames, -1)
    for frame_number in range(no_frames):
        zig_zag_vectors_frame = zig_zag_vectors_video[frame_number]

        # reconstruim frame-ul pe fiecare channel
        no_channels = video_shape[-1]
        no_elements_zig_zag_vectors_on_frame = len(zig_zag_vectors_frame)

        if no_elements_zig_zag_vectors_on_frame % no_channels != 0:
            raise ValueError('the number of zig zag vectors does not match the number of channels!')

        zig_zag_vectors_ycbcr_image = np.array(zig_zag_vectors_frame).reshape(no_channels, -1)

        # generam templat-ul imaginii YCBCR
        ycbcr_image = np.zeros(video_shape[1:])

        for channel_number in range(no_channels):
            zig_zag_vectors_channel = zig_zag_vectors_ycbcr_image[channel_number]

            block_shape = quantization_matrix.shape
            no_elements_zig_zag_vectors = len(zig_zag_vectors_channel)
            no_elements_zig_zag_vector_per_block = math.prod(block_shape)

            if no_elements_zig_zag_vectors % no_elements_zig_zag_vector_per_block != 0:
                raise ValueError('the number of zig-zaz vectors does not match the block shape!')

            zig_zag_vectors_image_blocks = zig_zag_vectors_channel.reshape(-1, no_elements_zig_zag_vector_per_block)

            # transformarea vectorilor zig-zag in matricile corespunzatoare (matrici cuantizate)
            quant_matrix_image_blocks = np.array(
                [generate_matrix_from_zig_zag_vector(zig_zag_vector, block_shape) for zig_zag_vector in
                 zig_zag_vectors_image_blocks])

            # decuantizarea blocurilor
            dequant_image_blocks = np.multiply(quant_matrix_image_blocks, quantization_matrix)

            # aplicam inversa transformatei cosinus discrete in 2D
            image_blocks = np.array([scipy.fft.idctn(dequant_block) for dequant_block in dequant_image_blocks])

            # reconstruim imaginea din blocuri
            image = generate_image_from_blocks(image_blocks, video_shape[1:-1], block_shape)

            # normalizarea pixelilor in intervalul [0,255] pentru fiecare block
            image = Image.fromarray(image)

            # compunerea imaginii ycbcr
            ycbcr_image[:, :, channel_number] = image

        # facem conversia din formatul YCBCR la RGB
        rgb_image = skimage.color.ycbcr2rgb(ycbcr_image)

        # compunem video-ul
        video[frame_number, :, :, :] = rgb_image

    return video


def compress_video_using_jpeg_format(video, quantization_matrix, output_dir):
    # validarea input-urilor
    if not (type(video) is np.ndarray and video.ndim == 4):
        raise ValueError('parameter image must be a 2D numpy array!')
    if not (video.dtype == np.uint8 and np.min(video) == 0 and np.max(video) == 255):
        raise ValueError('parameter image must be a 2D numpy array of integers with values between 0 and 255!')
    if not (type(quantization_matrix) is np.ndarray and quantization_matrix.ndim == 2):
        raise ValueError('parameter quantization_matrix must be a 2D numpy array!')

    # aplicam algoritmul de compresie pe fiecare frame
    no_frames = video.shape[0]
    zig_zag_vectors_video = []
    for frame_number in range(no_frames):
        # obtinem imaginea RGB
        rgb_image = video[frame_number, :, :, :]

        # conversia imaginii din RGB YCBCR
        ycbcr_image = skimage.color.rgb2ycbcr(rgb_image)

        # aplicam algoritmul de compresie si obtinem vectorii zig-zag pe fiecare channel
        zig_zag_vectors_ycbcr_image = []
        no_channels = ycbcr_image.shape[-1]

        for channel_number in range(no_channels):
            # obtinem imaginea la nivel de channel
            image = ycbcr_image[:, :, channel_number]

            # scoatem media din semnal
            avg_pixel = calculate_average_pixel()
            # image = image - avg_pixel

            # impart imaginea gray scale in blocuri de 8x8
            image_blocks = np.array(generate_image_blocks(image))

            # aplicarea transformatei cosinus discreta (2D) pentru fiecare bloc
            dct_2d_image_blocks = np.array([scipy.fft.dctn(block) for block in image_blocks])

            # cuantizarea in frecventa utilizand matricea de cuantizare
            quant_image_blocks = np.round(np.divide(dct_2d_image_blocks, quantization_matrix)).astype(int)

            # obtinerea vectorilor zig-zag pentru fiecare block
            zig_zag_vector_image_blocks = [generate_zig_zag_vector_of_matrix(matrix) for matrix in quant_image_blocks]
            zig_zag_vectors_ycbcr_image.append(list(itertools.chain.from_iterable(zig_zag_vector_image_blocks)))

        zig_zag_vectors_video.append(list(itertools.chain.from_iterable(zig_zag_vectors_ycbcr_image)))

    zig_zag_vectors_video = list(itertools.chain.from_iterable(zig_zag_vectors_video))

    # encoding-ul utilizand algoritmul lui Huffman
    code_table, encoded_data = apply_huffman_encoding(zig_zag_vectors_video)

    # gruparea datelor intr-o singura structura
    compressed_video = {'huffman_code_table': code_table,
                        'huffman_encoded_data': encoded_data,
                        'video_shape': video.shape,
                        'quantization_matrix': quantization_matrix}

    # salvarea datelor
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, 'compressed_video.pickle'), 'wb')
    pickle.dump(compressed_video, file)
    file.close()

    return compressed_video


def read_video_file(file_name, new_frame_size=None):
    video_capture = cv2.VideoCapture(file_name)

    video_frames = []
    while True:
        return_status, video_frame = video_capture.read()

        if return_status is False:
            break

        if new_frame_size is not None:
            video_frame = cv2.resize(video_frame, (512, 512), interpolation=cv2.INTER_LINEAR)

        video_frames.append(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))

    return np.array(video_frames)


def decompress_rgb_image_using_jpeg_algorithm(compressed_image):
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

    # decoding-ul utilizand algoritmul lui Huffman
    decoded_data = apply_huffman_decoding(huffman_code_table, huffman_encoded_data)

    # generam templat-ul imaginii
    ycbcr_image = np.zeros(image_shape)

    # reconstruim imaginea pe fiecare channel
    no_channels = image_shape[-1]
    no_elements_decoded_data = len(decoded_data)

    if no_elements_decoded_data % no_channels != 0:
        raise ValueError('the number of encoded elements that not match the image number of channels!')

    zig_zag_vector_ycbcr_image = np.array(decoded_data).reshape(no_channels, -1)
    for channel_number in range(no_channels):
        zig_zag_vectors_on_channel = zig_zag_vector_ycbcr_image[channel_number]

        block_shape = quantization_matrix.shape
        no_elements_zig_zag_vectors = len(zig_zag_vectors_on_channel)
        no_elements_zig_zag_vector_per_block = math.prod(block_shape)

        if no_elements_zig_zag_vectors % no_elements_zig_zag_vector_per_block != 0:
            raise ValueError('the number of zig-zaz vectors does not match the block shape!')

        zig_zag_vectors_image_blocks = zig_zag_vectors_on_channel.reshape(-1, no_elements_zig_zag_vector_per_block)

        # transformarea vectorilor zig-zag in matricile corespunzatoare (matrici cuantizate)
        quant_matrix_image_blocks = np.array(
            [generate_matrix_from_zig_zag_vector(zig_zag_vector, block_shape) for zig_zag_vector in
             zig_zag_vectors_image_blocks])

        # decuantizarea blocurilor
        dequant_image_blocks = np.multiply(quant_matrix_image_blocks, quantization_matrix)

        # aplicam inversa transformatei cosinus discrete in 2D
        image_blocks = np.array([scipy.fft.idctn(dequant_block) for dequant_block in dequant_image_blocks])

        # reconstruim imaginea din blocuri
        image = generate_image_from_blocks(image_blocks, image_shape[:-1], block_shape)

        # compunerea imaginii ycbcr
        ycbcr_image[:, :, channel_number] = image

    # conversia imaginii din formatul YCBCR in formatul RGB
    rgb_image = skimage.color.ycbcr2rgb(ycbcr_image)
    # rgb_image = np.array(Image.fromarray(np.uint8(rgb_image)))

    return rgb_image


def compress_rgb_image_using_jpeg_algorithm(rgb_image, quantization_matrix, output_dir):
    # validarea input-urilor
    if not (type(rgb_image) is np.ndarray and rgb_image.ndim == 3):
        raise ValueError('parameter image must be a 2D numpy array!')
    if not (rgb_image.dtype == np.uint8 and np.min(rgb_image) == 0 and np.max(rgb_image) == 255):
        raise ValueError('parameter image must be a 2D numpy array of integers with values between 0 and 255!')
    if not (type(quantization_matrix) is np.ndarray and quantization_matrix.ndim == 2):
        raise ValueError('parameter quantization_matrix must be a 2D numpy array!')

    # conversia imaginii din RGB YCBCR
    ycbcr_image = skimage.color.rgb2ycbcr(rgb_image)

    # aplicam algoritmul de compresie si obtinem vectorii zig-zag pe fiecare channel
    zig_zag_vectors_ycbcr_image = []
    no_channels = ycbcr_image.shape[-1]
    for channel_number in range(no_channels):
        # obtinem imaginea la nivel de channel
        image = ycbcr_image[:, :, channel_number]

        # scoatem media din semnal
        avg_pixel = calculate_average_pixel()
        # image = image - avg_pixel

        # impart imaginea gray scale in blocuri de 8x8
        image_blocks = np.array(generate_image_blocks(image))

        # aplicarea transformatei cosinus discreta (2D) pentru fiecare bloc
        dct_2d_image_blocks = np.array([scipy.fft.dctn(block) for block in image_blocks])

        # cuantizarea in frecventa utilizand matricea de cuantizare
        quant_image_blocks = np.round(np.divide(dct_2d_image_blocks, quantization_matrix)).astype(int)

        # obtinerea vectorilor zig-zag pentru fiecare block
        zig_zag_vector_image_blocks = [generate_zig_zag_vector_of_matrix(matrix) for matrix in quant_image_blocks]
        zig_zag_vectors_ycbcr_image.append(list(itertools.chain.from_iterable(zig_zag_vector_image_blocks)))
    zig_zag_vectors_ycbcr_image = list(itertools.chain.from_iterable(zig_zag_vectors_ycbcr_image))

    # encoding-ul utilizand algoritmul lui Huffman
    code_table, encoded_data = apply_huffman_encoding(zig_zag_vectors_ycbcr_image)

    # gruparea datelor intr-o singura structura
    compressed_image = {'huffman_code_table': code_table,
                        'huffman_encoded_data': encoded_data,
                        'image_shape': rgb_image.shape,
                        'quantization_matrix': quantization_matrix}

    # salvarea datelor
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, 'compressed_image.pickle'), 'wb')
    pickle.dump(compressed_image, file)
    file.close()

    return compressed_image


def plot_original_vs_decompressed_image(original_image, decompressed_image, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data = {'Original image': original_image,
            'Decompressed image': decompressed_image}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    for ax, (plot_title, image) in zip(axes, data.items()):
        if original_image.ndim == 2 and decompressed_image.ndim == 2:
            ax.imshow(image, cmap='gray')
        elif original_image.ndim == 3 and decompressed_image.ndim == 3:
            ax.imshow(image)
        else:
            raise ValueError(
                'parameters original_image and decompressed_image should have the same number of dimensions and the number of dimensions must be 2 or 3!')
        ax.set_title(plot_title)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'original_vs_decompressed_image.png'))


def decompress_gray_scale_image_using_jpeg_format(compressed_image):
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
    code_table = compressed_image.get('huffman_code_table')
    encoded_data = compressed_image.get('huffman_encoded_data')
    image_shape = compressed_image.get('image_shape')

    # decoding-ul utilizand algoritmul lui Huffman
    decoded_data = apply_huffman_decoding(code_table, encoded_data)

    # salvarea vectorilor zig zag in liste separate
    block_shape = quantization_matrix.shape
    no_elemenets_zig_zag_vector = math.prod(block_shape)
    zig_zag_vectors_image_blocks = np.array(decoded_data).reshape(-1, no_elemenets_zig_zag_vector)

    # transformarea vectorilor zig-zag in matricile corespunzatoare (matrici cuantizate)
    quant_matrix_image_blocks = np.array(
        [generate_matrix_from_zig_zag_vector(zig_zag_vector, block_shape) for zig_zag_vector in
         zig_zag_vectors_image_blocks])

    # decuantizarea blocurilor
    dequant_image_blocks = np.multiply(quant_matrix_image_blocks, quantization_matrix)

    # aplicam inversa transformatei cosinus discrete in 2D
    image_blocks = np.array([scipy.fft.idctn(dequant_block) for dequant_block in dequant_image_blocks])

    # adaugam pixelul mediu
    avg_pixel = calculate_average_pixel()
    # image_blocks = np.array([image + avg_pixel for image in image_blocks])

    # reconstruim imaginea din blocuri
    image = generate_image_from_blocks(image_blocks, image_shape, block_shape)

    # normalizarea pixelilor in intervalul [0,255] pentru fiecare block
    image = np.array(Image.fromarray(np.uint8(image)))

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


def compress_gray_scale_image_using_jpeg_format(image, quantization_matrix, output_dir):
    # validarea input-urilor
    if not (type(image) is np.ndarray and image.ndim == 2):
        raise ValueError('parameter image must be a 2D numpy array!')
    if not (image.dtype == np.uint8 and np.min(image) == 0 and np.max(image) == 255):
        raise ValueError('parameter image must be a 2D numpy array of integers with values between 0 and 255!')
    if not (type(quantization_matrix) is np.ndarray and quantization_matrix.ndim == 2):
        raise ValueError('parameter quantization_matrix must be a 2D numpy array!')

    # impart imaginea gray scale in blocuri de 8x8
    image_blocks = np.array(generate_image_blocks(image))

    # scoatem media din semnal
    avg_pixel = calculate_average_pixel()
    # image_blocks = np.array([image - avg_pixel for image in image_blocks])

    # aplicarea transformatei cosinus discreta (2D) pentru fiecare bloc
    dct_2d_image_blocks = np.array([scipy.fft.dctn(block) for block in image_blocks])

    # cuantizarea in frecventa utilizand matricea de cuantizare
    quant_image_blocks = np.round(np.divide(dct_2d_image_blocks, quantization_matrix)).astype(int)

    # obtinerea vectorilor zig-zag pentru fiecare block
    zig_zag_vector_image_blocks = [generate_zig_zag_vector_of_matrix(matrix) for matrix in quant_image_blocks]
    zig_zag_vector_image_blocks = list(itertools.chain.from_iterable(zig_zag_vector_image_blocks))

    # encoding-ul utilizand algoritmul lui Huffman
    code_table, encoded_data = apply_huffman_encoding(zig_zag_vector_image_blocks)

    # gruparea datelor intr-o singura structura
    compressed_image = {'huffman_code_table': code_table,
                        'huffman_encoded_data': encoded_data,
                        'image_shape': image.shape,
                        'quantization_matrix': quantization_matrix}

    # salvarea datelor
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file = open(os.path.join(output_dir, 'compressed_image.pickle'), 'wb')
    pickle.dump(compressed_image, file)
    file.close()

    return compressed_image


def apply_huffman_encoding(data):
    huffmann_alhorithm = HuffmanCodec.from_data(data)
    encoded_data = huffmann_alhorithm.encode(data)
    code_table = huffmann_alhorithm.get_code_table()

    return code_table, encoded_data


def apply_huffman_decoding(code_tabel, encoded_data):
    return HuffmanCodec(code_tabel).decode(encoded_data)


def calculate_average_pixel():
    min_pixel = 0
    max_pixel = 255
    avg_pixel = np.mean([min_pixel, max_pixel])

    return np.round(avg_pixel).astype(int)


def generate_zig_zag_vector_of_matrix(matrix):
    '''
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

    '''
    # codul de mai sus este corect si functioneaza pentru orice matrice
    # dar este foarte incet
    # intrucat blocurile noaste sunt intotdeauna 8x8
    # voi face hard coding pentru indicii elementelor

    elements_on_diagonals_indices = np.array(all_elements_on_diagonals_indices())
    zig_zag_matrix_elements = matrix[elements_on_diagonals_indices[:, 0], elements_on_diagonals_indices[:, 1]]

    return zig_zag_matrix_elements.tolist()


def generate_matrix_from_zig_zag_vector(zig_zag_vector, block_shape):
    elements_on_diagonals_indices = np.array(all_elements_on_diagonals_indices())
    matrix = np.zeros((block_shape))
    matrix[elements_on_diagonals_indices[:, 0], elements_on_diagonals_indices[:, 1]] = zig_zag_vector

    return matrix


def all_elements_on_diagonals_indices():
    return [(0, 0),
            (0, 1), (1, 0),
            (2, 0), (1, 1), (0, 2),
            (0, 3), (1, 2), (2, 1), (3, 0),
            (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
            (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0),
            (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
            (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0),
            (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
            (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2),
            (7, 3), (6, 4), (5, 5), (4, 6), (3, 7),
            (4, 7), (5, 6), (6, 5), (7, 4),
            (7, 5), (6, 6), (5, 7),
            (6, 7), (7, 6),
            (7, 7)]


def generate_image_blocks(image, block_size=(8, 8)):
    if not (type(block_size) is tuple and len(block_size) == 2):
        raise ValueError('parameter block_size must be tuple of two elements!')

    block_no_rows, block_no_cols = block_size
    if not (type(block_no_rows) is int and block_no_rows > 0 and type(block_no_cols) is int and block_no_cols > 0):
        raise ValueError('the components of the block_size should be strictly positive integers!')

    if not (type(image) is np.ndarray and image.ndim == 2):
        raise ValueError('the parameter image should be a 2D numpy array!')

    image_no_rows, image_no_cols = image.shape
    if image_no_rows % block_no_rows != 0:
        raise ValueError('the image number of rows should be a a multiple of the block number of rows!')
    if image_no_cols % block_no_cols != 0:
        raise ValueError('the image number of columns should be a a multiple of the block number of columns!')

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
