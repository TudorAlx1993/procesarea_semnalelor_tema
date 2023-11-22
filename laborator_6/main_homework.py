import os
import itertools
import scipy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    output_dir = './outputs_homework'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ex_1(100, output_dir)
    ex_2(25, output_dir)
    ex_3(output_dir)
    ex_4('./inputs/train.csv', output_dir)


def ex_4(file_name, output_dir):
    # reading and formatting the input data
    data = pd.read_csv(file_name)
    data.columns = map(str.upper, data.columns)
    data.DATETIME = pd.to_datetime(data.DATETIME, format='%d-%m-%Y %H:%M')
    data.sort_values(by='DATETIME', inplace=True)

    # ma asigur ca esantionaele sunt generate la intervale egale
    hours_between_samples = np.unique(np.diff(data.DATETIME.values).astype('timedelta64[h]'))
    assert len(hours_between_samples) == 1

    # punctul a)
    # filtram datele astfel incat sa incepem cu esantionul numarul 15000
    start_index = 15000
    end_index = data.shape[0]
    data = data[data.index.isin(range(start_index, end_index + 1))]

    # filtram datele astfel incat sa primul esantion sa inceapa cu ora 00:00
    start_index = data.index.values[0]
    while True:
        if pd.to_datetime(data.filter(items=[start_index], axis=0).DATETIME.values[0]).hour == 0:
            break
        start_index += 1
    end_index = data.index.values[-1]
    data = data[data.index.isin(range(start_index, end_index + 1))]

    # filtram datele astfel incat sa acoperim o perioada de 3 zile
    start_time = data.DATETIME.values[0]
    end_time = start_time + pd.DateOffset(days=3) - pd.Timedelta(hours=1)
    data = data[(data.DATETIME >= start_time) & (data.DATETIME <= end_time)]

    samples = data.index.values
    signal = data.COUNT.values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(samples, signal, linewidth='3.5')
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Count')
    ax.set_title(
        'Evolution of traffic data between {} and {}'.format(pd.to_datetime(start_time), pd.to_datetime(end_time)))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_a.png'))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_a.pdf'), format='pdf')

    # punctul b)
    window_lengths = np.array([5, 9, 13, 17])
    moving_averages = [np.convolve(signal, np.ones(window_length), 'valid') / window_length for window_length in
                       window_lengths]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(samples, signal, linewidth=3.5, label='Original signal')
    for window_length, moving_average in zip(window_lengths, moving_averages):
        no_points_moving_average = moving_average.shape[0]
        no_points_to_skip = len(signal) - no_points_moving_average
        ax.plot(samples[no_points_to_skip:], moving_average,
                linewidth=2.5,
                label='Smooth signal with w={}'.format(window_length))
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Count')
    ax.set_title('Signal smoothing using moving average')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_b.png'))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_b.pdf'), format='pdf')

    # punctul c)
    no_samples = data.shape[0]
    duration_hours = no_samples * 1
    duration_seconds = duration_hours * 3600
    sample_frequency = no_samples / duration_seconds  # adica 1/3600 (in Hz)

    X = np.fft.fft(signal)
    X_abs = np.abs(X / no_samples)
    X_abs = X_abs[:no_samples // 2]
    X_freqs = sample_frequency * np.linspace(0, no_samples // 2, no_samples // 2) / no_samples
    cutting_frequency = np.median(X_freqs)
    nyquist_frequency = sample_frequency / 2
    # regula de trei simpla
    # nyquist_frequency ........................ 1
    # cutting_frequency .........................x
    normalized_cutting_frequency = cutting_frequency * 1.0 / nyquist_frequency

    file_content = '\t* frecventa de taiere este mediana frecventelor, adica {} Hz\n'.format(cutting_frequency)
    file_content += '\t* am ales mediana frecventelor deoarece:\n'
    file_content += '\t\t* toate frecventele inalte vor fi peste mediana si implicit vor fi eliminate prin aceasta metoda\n'
    file_content += '\t\t* media nu este intotdeauna semnificativa din punct de vedere statistic\n'
    file_content += '\t\t* semnalul filtrat prin intermediul medianei va fi mult mai smooth\n'
    file_content += '\t* frecventa de taiere normalizata in intervalul [0,1] este {:.4f}\n'.format(
        normalized_cutting_frequency)

    file = open(os.path.join(output_dir, 'ex_4_pct_c.txt'), 'w')
    file.write(file_content)
    file.close()

    # punctul d)
    filter_order = 5
    attenuation_factor = 5

    b_butterworth_filter, a_butterworth_filter = scipy.signal.butter(filter_order, normalized_cutting_frequency,
                                                                     btype='low')
    b_chebyshev, a_chebyshev = scipy.signal.cheby1(filter_order, attenuation_factor, normalized_cutting_frequency,
                                                   btype='low')

    butterworth_filter_coefs = (b_butterworth_filter, a_butterworth_filter)
    chebyshev_filter_coefs = (b_chebyshev, a_chebyshev)
    filters = {'butterworth': butterworth_filter_coefs, 'chebyshev': chebyshev_filter_coefs}
    file = open(os.path.join(output_dir, 'ex_4_pct_d_filter_coefs.npy'), 'wb')
    pickle.dump(filters, file)
    file.close()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    for index, filter_name in enumerate(filters.keys()):
        b_coefs, a_coefs = filters[filter_name]
        w, h = scipy.signal.freqz(b_coefs, a_coefs)
        axes[index].plot(w, 20 * np.log10(np.abs(h)), linewidth=5.0)
        axes[index].set_xlabel('Frequency (radians/second)')
        axes[index].set_ylabel('Amplitude (db)')
        axes[index].set_title('{} filter'.format(filter_name.capitalize()))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_d_filters.png'))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_d_filters.pdf'), format='pdf')

    # punctul e)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(samples, signal, linewidth=4.0, label='Original signal')
    for filter_name in filters.keys():
        b_coefs, a_coefs = filters[filter_name]
        filtered_signal = scipy.signal.filtfilt(b_coefs, a_coefs, signal)
        ax.plot(samples, filtered_signal, linewidth=4.0,
                label='Signal filteer with {}'.format(filter_name.capitalize()))
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Count')
    ax.set_title('Traffic data')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_e_signal_with_filters.png'))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_e_signal_with_filters.pdf'), format='pdf')

    file = open(os.path.join(output_dir, 'ex_4_pct_e_chosen_filter.txt'), 'w')
    file.write(
        'Aleg filtrul Chebyshev deoarece reduce foarte mult din amplitudinea semnalului original, adica scapam de componentele de frecventa inalta.\nAstfel putem observa mai usor pattern-uri specifice anumitor ore dintr-o zi.\n ')
    file.close()

    # punctul f)

    # pentru filtrul Butterworth
    filter_orders = [3, 5, 7]
    butterworth_filters = [scipy.signal.butter(filter_order, normalized_cutting_frequency, btype='low') for filter_order
                           in filter_orders]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(samples, signal, linewidth=4.0, label='Original signal')
    for index, (b_coefs, a_coefs) in enumerate(butterworth_filters):
        filtered_signal = scipy.signal.filtfilt(b_coefs, a_coefs, signal)
        ax.plot(samples, filtered_signal, linewidth=4.0,
                label='Filter order={}'.format(filter_orders[index]))
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Count')
    ax.set_title('Traffic data with Butterworth filter')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_f_butterworth_filter.png'))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_f_butterworth_filter.pdf'), format='pdf')

    file_content = 'Alegerea parametrilor:\n'
    file_content += '\t* filtrul Batterworth:\n'
    file_content += '\t\t* nu sunt diferente semnificative intre semnalele filtrate atunci cand ordinul filtrului este 3, 5 sau 7\n'

    # pentru filtrul Chebyshev
    attenuation_factors = [3, 5, 7]
    chebyshev_filters = [
        (scipy.signal.cheby1(filter_order, attenuation_factor, normalized_cutting_frequency, btype='low'), filter_order,
         attenuation_factor) for
        filter_order, attenuation_factor in itertools.product(filter_orders, attenuation_factors)]
    fig, ax = plt.subplots(figsize=(10, 20))
    ax.plot(samples, signal, linewidth=4.0, label='Original signal')
    for (b_coefs, a_coefs), filter_order, attenuation_factor in chebyshev_filters:
        filtered_signal = scipy.signal.filtfilt(b_coefs, a_coefs, signal)
        ax.plot(samples, filtered_signal, linewidth=4.0,
                label='N={} rp={}'.format(filter_order, attenuation_factor))
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Count')
    ax.set_title('Traffic data with Chebyshev filter')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_f_chebyshev_filter.png'))
    fig.savefig(os.path.join(output_dir, 'ex_4_pct_f_chebyshev_filter.pdf'), format='pdf')

    file_content += '\t* filtrul Chebyshev:\n'
    file_content += '\t\t* am facut produsul cartezian intre multimile [3,5,7] (pentru ordinul filtrului) si [3,5,7] (pentru factorul de atenuare)\n'
    file_content += '\t\t* am ales N=7, rp=7 ca parametri optimi\n'

    file = open(os.path.join(output_dir, 'ex_4_pct_f_chosen_paramters.txt'), 'w')
    file.write(file_content)
    file.close()


def ex_3(output_dir):
    start_time = 0.0
    end_time = 0.05
    no_points = 10 ** 4
    window_size = 200

    frequency = 100
    amplitude = 1.0
    phase = 0.0

    generate_sin_date = lambda A, freq, time, phase: A * np.sin(2 * np.pi * freq * time + phase)

    n = np.linspace(start_time, end_time, window_size)
    discrete_signal = generate_sin_date(amplitude, frequency, n, phase)

    time = np.linspace(start_time, end_time, no_points)
    cont_signal = generate_sin_date(amplitude, frequency, time, phase)

    def generate_hanning_window(N):
        # impart la N-1 si nu la N pentru a ma verifica cu numpy
        return 0.5 * np.array([1 - np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])

    def generate_rectangular_window(N):
        return np.array([1.0 for _ in range(N)])

    discrete_rectangular_window = generate_rectangular_window(window_size)
    discrete_hanning_window = generate_hanning_window(window_size)

    assert np.allclose(discrete_hanning_window, np.hanning(window_size))

    discrete_windows = discrete_rectangular_window * discrete_hanning_window
    discrete_signal_with_windowing = discrete_signal * discrete_windows

    cont_rectangular_window = generate_rectangular_window(no_points)
    cont_hanning_window = generate_hanning_window(no_points)

    cont_windows = cont_rectangular_window * cont_hanning_window
    cont_signal_with_windowing = cont_signal * cont_windows

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    left_top_ax, left_center_ax, left_bottom_ax = axes[:, 0]
    right_top_ax, right_center_ax, right_bottom_ax = axes[:, 1]

    left_top_ax.plot(time, cont_signal, linewidth=3.5)
    left_top_ax.set_title('Original continuous signal')
    left_center_ax.plot(time, cont_windows, linewidth=3.5)
    left_center_ax.set_title('Rectangular and Hanning continuous windows')
    left_bottom_ax.plot(time, cont_signal_with_windowing, linewidth=3.5)
    left_bottom_ax.set_title('Continuous signal with rectangular and Hanning windows')
    right_top_ax.stem(n, discrete_signal)
    right_top_ax.set_title('Original discrete signal')
    right_center_ax.stem(n, discrete_windows)
    right_center_ax.set_title('Rectangular and Hanning discrete windows')
    right_bottom_ax.stem(n, discrete_signal_with_windowing)
    right_bottom_ax.set_title('Discrete signal with rectangular and Hanning windows')
    for ax in axes.flatten():
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ex_3.png'))
    fig.savefig(os.path.join(output_dir, 'ex_3.pdf'), format='pdf')


def compute_1d_convolution(x, y):
    if not (type(x) == type(y) and type(x) is np.ndarray):
        raise ValueError('x and y parameters should be numpy arrays!')

    if not (x.ndim == 1 and y.ndim == 1):
        raise ValueError('x and y paramters should be 1D numpy arrays!')

    if not x.shape[0] == y.shape[0]:
        raise ValueError('x and y parameters should have the same number of elements!')

    no_elements = x.shape[0]
    convolution = np.array([np.sum(
        [y[second_index] * x[first_index - second_index] for second_index in range(no_elements) if
         first_index >= second_index]) for first_index in range(no_elements)])

    convolution = np.append(convolution, convolution[::-1])

    return convolution


def ex_1(n, output_dir):
    # simulez din distributia uniforma pe intervalul [0,1)
    x = np.random.rand(n)

    no_iterations = 3
    results = [('original signal (random from uniform distribution over the [0,1) interval)', x)]
    for iteration in range(1, no_iterations + 1):
        # x = compute_1d_convolution(x, x)
        x = polynomial_multiplication_direct_method(x, x)
        # pentru scalare: axa OY in graficele de mai jos sa fie intre 0 si 1 si integrala (adica suma) sa fie 1
        # scalarea nu e obligatorie, e doar sa nu imi afiseze valori mari pe axa OY
        results.append(('iteration {}'.format(iteration), x / np.sum(x)))

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


def polynomial_multiplication_direct_method(x, y):
    if not (type(x) == type(y) and type(x) is np.ndarray):
        raise ValueError('x and y parameters should be numpy arrays!')

    if not (x.ndim == 1 and y.ndim == 1):
        raise ValueError('x and y paramters should be 1D numpy arrays!')

    if not x.shape[0] == y.shape[0]:
        raise ValueError('x and y parameters should have the same number of elements!')

    no_elements = x.shape[0]
    result_no_elements = 2 * no_elements - 1
    result = np.zeros(shape=result_no_elements)

    for first_index in range(no_elements):
        for second_index in range(no_elements):
            result[first_index + second_index] += x[first_index] * y[second_index]

    return result


def ex_2(N, output_dir):
    min_int = 1
    max_int = 100

    p = np.random.randint(low=min_int, high=max_int + 1, size=N)
    q = np.random.randint(low=min_int, high=max_int + 1, size=N)

    polynomial_multiplication_with_numpy = np.polymul(p, q)
    polynomial_multiplication_with_direct_method = polynomial_multiplication_direct_method(p, q)

    no_elements = p.shape[0]
    polynomial_multiplication_no_elements = 2 * no_elements - 1
    no_pads = polynomial_multiplication_no_elements - no_elements
    p_with_pad = np.pad(p, (0, no_pads))
    q_with_pad = np.pad(q, (0, no_pads))

    fourier_p = np.fft.fft(p_with_pad)
    fourier_q = np.fft.fft(q_with_pad)
    product_of_p_and_q_in_frequency = fourier_p * fourier_q
    polynomial_multiplication_with_fourier = np.fft.ifft(product_of_p_and_q_in_frequency)

    assert np.allclose(polynomial_multiplication_with_fourier.imag, 0.0)
    polynomial_multiplication_with_fourier = polynomial_multiplication_with_fourier.real

    assert np.allclose(polynomial_multiplication_with_fourier, polynomial_multiplication_with_numpy)
    assert np.allclose(polynomial_multiplication_with_fourier, polynomial_multiplication_with_direct_method)
    assert np.allclose(polynomial_multiplication_with_numpy, polynomial_multiplication_with_direct_method)

    file_content = 'Polynom p: {}\n'.format(p)
    file_content += 'Polynom q: {}\n'.format(q)
    file_content += 'Polynomial multiplication with numpy: {}\n'.format(polynomial_multiplication_with_numpy)
    file_content += 'Polynomial multipilication with Fourier: {}\n'.format(polynomial_multiplication_with_fourier)
    file_content += 'Polynomial multiplication with direct method: {}\n'.format(
        polynomial_multiplication_with_direct_method)

    file = open(os.path.join(output_dir, 'ex_2.txt'), 'w')
    file.write(file_content)
    file.close()


if __name__ == '__main__':
    main()
