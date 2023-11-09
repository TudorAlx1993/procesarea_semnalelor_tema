import os
import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from scipy.io import wavfile


def main():
    output_dir = './outputs_homework'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exercise_1(output_dir)
    exercise_2(output_dir)
    exercise_3(output_dir)
    exercise_4(output_dir)
    exercise_5()
    exercise_6('./inputs/vowels.wav', output_dir)
    exercise_7(output_dir)


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
    # pentru rezolvarea acestui exercitiu am folosit informatiile din cursul nr 4

    generate_sin_signal = lambda amplitude, t, frequency, phase: amplitude * np.sin(2 * np.pi * t * frequency + phase)

    amplitude = 1.0
    phase = 0.0
    start = 0.0
    stop = 1.0

    frequency_sampling = 9
    n = np.linspace(start, stop, frequency_sampling + 1)

    no_points_cont = 10 ** 3
    t = np.linspace(start, stop, no_points_cont)

    # continuous signals
    frequency_first_signal = 20
    cont_first_signal = generate_sin_signal(amplitude, t, frequency_first_signal, phase)
    frequency_second_signal = frequency_first_signal - 1 * frequency_sampling
    cont_second_signal = generate_sin_signal(amplitude, t, frequency_second_signal, phase)
    frequency_third_signal = frequency_first_signal - 2 * frequency_sampling
    cont_third_signal = generate_sin_signal(amplitude, t, frequency_third_signal, phase)

    # discrete signals
    discrete_first_signal = generate_sin_signal(amplitude, n, frequency_first_signal, phase)
    discrete_second_signal = generate_sin_signal(amplitude, n, frequency_second_signal, phase)
    discrete_third_signal = generate_sin_signal(amplitude, n, frequency_third_signal, phase)

    assert np.allclose(discrete_first_signal, discrete_second_signal)
    assert np.allclose(discrete_second_signal, discrete_third_signal)
    assert np.allclose(discrete_third_signal, discrete_first_signal)

    colors = ['lightblue', 'lightblue', 'tomato', 'lime']
    cont_signal_frequencies = [frequency_first_signal, frequency_first_signal, frequency_second_signal,
                               frequency_third_signal]
    cont_signals = [cont_first_signal, cont_first_signal, cont_second_signal, cont_third_signal]
    discrete_signals = [discrete_first_signal, discrete_second_signal, discrete_third_signal]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
    for ax, color, cont_signal, cont_signal_freq in zip(axes, colors, cont_signals, cont_signal_frequencies):
        ax.plot(t, cont_signal, color=color, linewidth=2.5)
        ax.set_title('Continuous signal frequency = {} Hz'.format(cont_signal_freq))
        ax.set_xlabel('Time (t/n)')
        ax.set_ylabel('Value')
    for ax, discret_signal in zip(axes[1:], discrete_signals):
        ax.scatter(n, discret_signal, color='darkviolet', alpha=1.0, linewidth=3.5)
    fig.suptitle('Sampling frequency = {} Hz'.format(frequency_sampling), x=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_2.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_2.pdf'), format='pdf')


def exercise_3(output_dir):
    generate_sin_signal = lambda amplitude, t, frequency, phase: amplitude * np.sin(2 * np.pi * t * frequency + phase)

    amplitude = 1.0
    phase = 0.0
    start = 0.0
    stop = 1.0

    frequency_sampling = 14
    n = np.linspace(start, stop, frequency_sampling + 1)

    no_points_cont = 10 ** 3
    t = np.linspace(start, stop, no_points_cont)

    # continuous signals
    frequency_first_signal = 6
    cont_first_signal = generate_sin_signal(amplitude, t, frequency_first_signal, phase)
    frequency_second_signal = 5
    cont_second_signal = generate_sin_signal(amplitude, t, frequency_second_signal, phase)
    frequency_third_signal = 3
    cont_third_signal = generate_sin_signal(amplitude, t, frequency_third_signal, phase)

    # discrete signals
    discrete_first_signal = generate_sin_signal(amplitude, n, frequency_first_signal, phase)
    discrete_second_signal = generate_sin_signal(amplitude, n, frequency_second_signal, phase)
    discrete_third_signal = generate_sin_signal(amplitude, n, frequency_third_signal, phase)

    assert not np.allclose(discrete_first_signal, discrete_second_signal)
    assert not np.allclose(discrete_second_signal, discrete_third_signal)
    assert not np.allclose(discrete_third_signal, discrete_first_signal)

    colors = ['lightblue', 'lightblue', 'tomato', 'lime']
    cont_signal_frequencies = [frequency_first_signal, frequency_first_signal, frequency_second_signal,
                               frequency_third_signal]
    cont_signals = [cont_first_signal, cont_first_signal, cont_second_signal, cont_third_signal]
    discrete_signals = [discrete_first_signal, discrete_second_signal, discrete_third_signal]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
    for ax, color, cont_signal, cont_signal_freq in zip(axes, colors, cont_signals, cont_signal_frequencies):
        ax.plot(t, cont_signal, color=color, linewidth=2.5)
        ax.set_title('Continuous signal frequency = {} Hz'.format(cont_signal_freq))
        ax.set_xlabel('Time (t/n)')
        ax.set_ylabel('Value')
    for ax, discret_signal in zip(axes[1:], discrete_signals):
        ax.scatter(n, discret_signal, color='darkviolet', alpha=1.0, linewidth=3.5)
    fig.suptitle('Sampling frequency = {} Hz'.format(frequency_sampling), x=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercise_3.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_3.pdf'), format='pdf')


def exercise_4(output_dir):
    min_freq = 40
    max_freq = 200

    file_content = 'Contrabas:\n' + \
                   '\t * frecventa minima: {} Hz\n'.format(min_freq) + \
                   '\t * frecventa maxima: {} Hz\n'.format(max_freq) + \
                   'Frecventa minima de esantionare trebuie sa fie strict mai mare ca {} Hz pentru a nu exista pierderi de informatie.\n'.format(
                       2 * max_freq)

    file = open(os.path.join(output_dir, 'exercise_4.txt'), 'w')
    file.write(file_content)
    file.close()


def exercise_5():
    # a se vedea fisierul exercise_5.png directorul ./outputs_homework
    # fisierul reprezinta un screenshot din Audacity cu inregistrarea vocalelor
    # vocalele se pot distinge pe baza instensitatii semnalului pentru fiecare vocala si a duratei acesteia
    # de asemenea, vocalele pot fi identificate si prin analiza de frecventa prin intermediul spectogramei
    pass


def exercise_6(file_name, output_dir):
    # pe telefon inregistrarea este in format .m4a
    # am reexportat-o in format .wav cu ajutorul Audacity

    rate, signal = wavfile.read(file_name)

    n = signal.shape[0]
    grouping_ratio = 0.002
    overlap_ratio = 0.5
    no_elements_per_group = int(grouping_ratio * n)
    no_overlaps = int(overlap_ratio * no_elements_per_group)
    groups_of_signals = [signal[index:index + no_elements_per_group] for index in
                         range(0, n - no_elements_per_group, no_overlaps)]
    groups_of_signals_abs_fft = np.array(
        [np.log10(np.abs(np.fft.fft(signal)[:(signal.shape[0] // 2)])) for signal in groups_of_signals]).transpose()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(groups_of_signals_abs_fft)
    ax.set_xlabel('Time')
    ax.set_ylabel('Log10 of Absolute Frequency')
    ax.set_title('Spectogram of vowels')
    fig.savefig(os.path.join(output_dir, 'exercise_6.png'))
    fig.savefig(os.path.join(output_dir, 'exercise_6.pdf'), format='pdf')


def exercise_7(output_dir):
    signal_power = 90
    snr_db = 80

    # SNR_dB = 10 log10 SNR
    # SNR_db/10=log10 SNR
    # SNR=10^(SNR_db/10)
    snr = np.power(10, (snr_db / 10))

    noise_power = signal_power / snr

    def calculate_snr_db(signal_power, noise_power):
        snr = signal_power / noise_power
        return 10 * np.log10(snr)

    assert calculate_snr_db(signal_power, noise_power) == snr_db

    file_content = 'Signal power = {} dB\nSNR_DB = {} dB\nNoise power = {} dB\n'.format(signal_power, snr_db,
                                                                                        noise_power)
    file = open(os.path.join(output_dir, 'exercise_7.txt'), 'w')
    file.write(file_content)
    file.close()


if __name__ == '__main__':
    main()
