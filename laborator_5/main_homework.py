import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def main():
    output_dir = './outputs_homework'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exercise_1('./inputs/train.csv', output_dir)


def exercise_1(file_name, output_dir):
    # citim si formatarea datelor
    data = pd.read_csv(file_name)
    data.columns = map(str.upper, data.columns)
    data.DATETIME = pd.to_datetime(data.DATETIME, format='%d-%m-%Y %H:%M')
    data.sort_values(by='DATETIME', inplace=True)

    # raspunsurile le scriu in fisier
    file_content = 'Exercitiul 1\n\n'

    # punctul a)
    file_content += 'Punctul a):\n'
    hours_between_samples = np.unique(np.diff(data.DATETIME.values).astype('timedelta64[h]'))
    # ma asigur ca nu exista pauze de timp intre masuratori
    assert len(hours_between_samples) == 1
    hours_between_samples = hours_between_samples[0].astype(int)
    file_content += '\t* inregistrarile din fisierul de input apar la intervale de {} ore\n'.format(
        hours_between_samples)

    no_samples = data.shape[0]
    period_in_days = pd.Timedelta(data.DATETIME[no_samples - 1] - data.DATETIME[0] + pd.Timedelta(1, 'h')).days
    # time_per_sample_in_days = 1 / 24
    # sample_frequency = np.power((no_samples / period_in_days) / time_per_sample_in_days, -1)
    # aici aplic formula din curs cu unitatile de masura de timp in secunde
    # period_in_days*24*3600: aici transform timpul din zile in secunde
    # esantioanele vin din ora in ora, adica o data da 3600 de secunde
    # sample_frequency = np.power((no_samples / (period_in_days * 24 * 3600)) * (3600), -1)
    duration_hours = no_samples * 1
    duration_seconds = duration_hours * 3600
    sample_frequency = no_samples / duration_seconds
    file_content += '\t* frecventa de esantionare este {} Hz\n'.format(sample_frequency)

    # punctul b)
    file_content += 'Punctul b):\n'
    first_sample_date = data.DATETIME[0]
    last_sample_date = data.DATETIME[no_samples - 1]
    file_content += '\t* intervalul de timp al esantioanelor este cuprins intre {} si {}\n'.format(first_sample_date,
                                                                                                   last_sample_date)

    # punctul c)
    file_content += 'Punctul c):\n'
    max_frequency = sample_frequency / 2
    file_content += '\t* frecventa maxima prezenta in semnal este maxim 50% din frecventa de esantionare, adica cel mult {} Hz\n'.format(
        max_frequency)

    # punctul d)
    file_content += 'Punctul d):\n'
    file_content += '\t* a se vedea fisierul ex_1_d.png din directorul {}\n'.format(output_dir)

    x = data.COUNT.values

    X = np.fft.fft(x)
    X_abs = np.abs(X / no_samples)
    X_abs = X_abs[:no_samples // 2]
    X_freqs = sample_frequency * np.linspace(0, no_samples // 2, no_samples // 2) / no_samples

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.stem(X_freqs, X_abs)
    ax.set_title('Absolute value of the Fourier Transform')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Absolute value')
    fig.savefig(os.path.join(output_dir, 'ex_1_d.png'))
    fig.savefig(os.path.join(output_dir, 'ex_1_d.pdf'), format='pdf')

    # punctul e)
    file_content += 'Punctul e)\n'
    # X_abs[0] reprezinta modulul transformatei Fourier pentru frecventa 0
    assert X_freqs[0] == 0
    file_content += '\t* modulul transformatei Fourier pentru frecventa {} Hz este {:.4f}\n'.format(X_freqs[0],
                                                                                                    X_abs[0])
    if not np.allclose(X_abs[0], 0):
        file_content += '\t* deci exista o componenta continua in cadrul semnalului\n'
    else:
        file_content += '\t* deci nu exista o componenta continua in cadrul semnalului\n'
    file_content += '\t* a se vedea fisierul ex_1_e.png din directorul {} pentru a vedea semnalul fara componenta continua\n'.format(
        output_dir)

    assert not np.allclose(X_abs[0], 0)
    X_without_0_freq = X.copy()
    X_without_0_freq[0] = 0.0 + 0.0j
    x_wihout_cc_component = np.fft.ifft(X_without_0_freq)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(range(1, no_samples + 1), x, label='Signal with cc', color='red')
    ax.plot(range(1, no_samples + 1), x_wihout_cc_component, label='Signal without cc', color='blue')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Signal')
    ax.set_title('Signal with and without the cc')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'ex_1_e.png'))
    fig.savefig(os.path.join(output_dir, 'ex_1_e.pdf'), format='pdf')

    # punctul f)
    file_content += 'Punctul f):\n'
    file_content += '\t* pentru acest subpunct nu voi lua in calcul frecventa 0 (cea aferenta componentei continue)\n'
    no_dominant_freqs = 4
    file_content += '\t* primele {} frecvente dominante in ordinea inversa a modulului transformatei Fourier sunt:\n'.format(
        no_dominant_freqs)

    X_abs_sort_desc_indexes = X_abs.argsort()[::-1]
    freqs_most_dominant = X_freqs[X_abs_sort_desc_indexes[1:][:no_dominant_freqs]]
    X_abs_most_dominant = X_abs[X_abs_sort_desc_indexes[1:][:no_dominant_freqs]]

    for freq, amplitude in zip(freqs_most_dominant, X_abs_most_dominant):
        file_content += '\t\t* frecventa {} Hz are modulul transformatei Fourier {}\n'.format(
            freq, amplitude)

    # punctul g)
    file_content += 'Punctul g):\n'
    file_content += '\t* a se vedea fisierul ex_1_g.png din directorul {}\n'.format(output_dir)

    data_all = data.copy()

    # filtram datele astfel incat sa incepem cu esantionul numarul 1000
    start_index = 1000
    end_index = data.shape[0]
    data = data[data.index.isin(range(start_index, end_index + 1))]

    # filtram datele astfel incat sa primul esantion sa fie aferent unei zile de luni
    start_index = data.index.values[0]
    while True:
        if pd.to_datetime(data.filter(items=[start_index], axis=0).DATETIME.values[0]).day_name() == 'Monday':
            break
        start_index += 1
    end_index = data.index.values[-1]
    data = data[data.index.isin(range(start_index, end_index + 1))]

    # filtram datele astfel incat sa acoperim o luna de esantioane
    start_time = data.DATETIME.values[0]
    end_time = start_time + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    data = data[(data.DATETIME >= start_time) & (data.DATETIME <= end_time)]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(data.DATETIME, data.COUNT, linewidth=3.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('Traffic evolution between {} and {}'.format(pd.to_datetime(data.DATETIME.values[0]),
                                                              pd.to_datetime(data.DATETIME.values[-1])))
    fig.savefig(os.path.join(output_dir, 'ex_1_g.png'))
    fig.savefig(os.path.join(output_dir, 'ex_1_g.pdf'), format='pdf')

    # punctul h)
    file_content += 'Punctul h):\n'

    # punctul i)
    file_content += 'Punctul i):\n'

    # scrierea rezultatelor in fisierul de output
    output_file = open(os.path.join(output_dir, 'exercise_1.txt'), 'w')
    output_file.write(file_content)
    output_file.close()


if __name__ == '__main__':
    main()
