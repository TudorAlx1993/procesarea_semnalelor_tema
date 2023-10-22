import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def main():
    # plot-urile si raspunsurile vor fi salvate in directorul outputs
    output_dir = './outputs'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exercise_1(output_dir)
    exercise_2(output_dir)
    exercise_3(output_dir)


def exercise_1(output_dir):
    # punctul a)
    min = 0.0
    max = 0.03
    step_cont = 0.0005
    # stop=max+step_cont deoarece in cerinta apare interval inchis
    t = np.arange(start=min, stop=max + step_cont, step=step_cont)

    # punctul b)
    function_x = lambda t: np.cos(520 * np.pi * t + np.pi / 3)
    function_y = lambda t: np.cos(280 * np.pi * t - np.pi / 3)
    function_z = lambda t: np.cos(120 * np.pi * t + np.pi / 3)

    results_cont = [
        {'series': function_x(t), 'function_name': '$x(t)=\cos(520 \pi t+\pi/3)$', 'function_symbol': '$x(t)$'},
        {'series': function_y(t), 'function_name': '$y(t)=\cos(280 \pi t-\pi/3)$', 'function_symbol': '$y(t)$'},
        {'series': function_z(t), 'function_name': '$z(t)=\cos(120 \pi t+\pi/3)$', 'function_symbol': '$z(t)$'}]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    for result, ax in zip(results_cont, axes):
        ax.plot(t, result['series'])
        ax.set_title(result['function_name'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercitiul_1_punctul_b.png'))

    # punctul c)
    frequency = 200
    step_discrete = 1 / frequency
    n = np.arange(start=min, stop=max + step_discrete, step=step_discrete)

    results_discrete = [{'series': function_x(n), 'function_symbol': '$x[n]$'},
                        {'series': function_y(n), 'function_symbol': '$y[n]$'},
                        {'series': function_z(n), 'function_symbol': '$z[n]$'}]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    for result_cont, result_discrete, ax in zip(results_cont, results_discrete, axes):
        ax.plot(t, result_cont['series'], label=result_cont['function_symbol'])
        ax.stem(n, result_discrete['series'], label=result_discrete['function_symbol'])
        ax.set_title(result_cont['function_name'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exercitiul_1_punctul_c.png'))


def exercise_2(output_dir):
    # punctul a)
    amplitude = 1.0
    phase = 0
    frequency = 400
    count = 1600
    start = 0.0
    # intrucat nu se spune nimic despre momentul in care se termina semnalul in cerinta
    # voi folosi intormatia din laborator referitoare la relatia dintre frecventa si periada semnalului
    end = 1 / frequency
    t = np.linspace(start=start, stop=end, num=count)
    x = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t, x, linewidth=4.0, color='red')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.set_title('Sinusoidal signal at a frequency of {} hertz'.format(frequency))
    fig.savefig(os.path.join(output_dir, 'exercitiul_2_punctul_a.png'))

    # punctul b)
    amplitude = 1.0
    phase = 0.0
    frequency = 800
    start = 0.0
    end = 3.0
    # aici am folosit formula din cursul 1
    # adica numarul total de masuratori
    count = int((end - start) * frequency)
    t = np.linspace(start=start, stop=end, num=count)
    x = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t, x, linewidth=4.0, color='red')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.set_title('Sinusoidal signal at a frequency of {} hertz that lasts {:.2f} seconds'.format(frequency, end))
    fig.savefig(os.path.join(output_dir, 'exercitiul_2_punctul_b.png'))

    # punctul d)
    start = 0.0
    frequency = 300
    end = 1 / frequency
    count = 1000
    t = np.linspace(start=start, stop=end, num=count)
    # aici am folosit definitia semnalului de pe wikipedia: https://en.wikipedia.org/wiki/Square_wave
    # adica x(t)=sgn(sin(2*pi*f*t))
    x = np.sign(np.sin(2 * np.pi * frequency * t))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t, x, linewidth=4.0, color='red')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.set_title('Square signal at a frequency of {} hertz'.format(frequency))
    fig.savefig(os.path.join(output_dir, 'exercitiul_2_punctul_d.png'))

    # punctul c)
    start = 0.0
    frequency = 240
    end = 1 / frequency
    count = 100
    t = np.linspace(start=start, stop=4 * end, num=count)
    # nu am reusit sa generez semnalul utilizand functiile np.floor sau np.mod
    # pe google am gasit documentatia functiei sawtooth din scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sawtooth.html
    # doar aici am vazut un exemplu in care acest semnal poate fi generat in functie de frecventa dorita
    x = signal.sawtooth(2 * np.pi * frequency * t)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(t, x, linewidth=4.0, color='red')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.set_title('Sawtooth signal at a frequency of {} hertz'.format(frequency))
    fig.savefig(os.path.join(output_dir, 'exercitiul_2_punctul_c.png'))

    # punctul e)
    n = 128
    random_matrix = np.random.rand(n, n)

    fig, ax = plt.subplots(figsize=(10, 10))
    # am facut-o grayscale deoarece imaginele color au trei dimensiuni (adica cub si nu matrice cum am generat-o eu)
    ax.imshow(random_matrix, cmap='gray')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Random image based on a uniform distribution over $[0,1)$ interval')
    fig.savefig(os.path.join(output_dir, 'exercitiul_2_punctul_e.png'))

    # punctul f)
    n = 128
    '''
    Algoritmul de generara a matricii:
        * pas 1:  - generam o matrice in care fiecare linie este un vector de 0 sau un vector de 1
                  - daca indicele liniei este par atunci vectorul este generat cu np.zeros(n)
                  - daca indicele liniei este impar atunci vectorul este generat cu np.ones(n)
                  - numaram incepand cu 0 (adica indicele primei linii este 0)
        * pas 2:  generam transpusa matricei de la pasul anterior
        * pas 3:  pe fiecare coloana calculam suma cumulativa
    '''
    matrix = np.array([np.zeros(shape=n) if row_index % 2 == 0 else np.ones(shape=n) for row_index in range(n)])
    matrix = matrix.transpose()
    for col_index in range(matrix.shape[1]):
        matrix[:, col_index] = np.cumsum(matrix[:, col_index])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(matrix, cmap='gray')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('My image')
    fig.savefig(os.path.join(output_dir, 'exercitiul_2_punctul_f.png'))


def exercise_3(output_dir):
    frequency = 2000

    # punctul a)
    time_step = 1 / frequency
    message = 'a) Intervalul de timp intre 2 esantioane este {} secunde.\n'.format(time_step)

    # punctul b)
    # time_step este 0.0005 secunde
    '''
    0.0005 secunde ....... 1 esantion ..... 4 biti  
    3600 secunde .......... x esantionae ..... y biti 
    '''
    seconds_per_hour = 3600
    no_samples_per_hour = seconds_per_hour / time_step
    bites_per_sample = 4
    bites_per_hour = bites_per_sample * no_samples_per_hour
    bites_per_byte = 8
    bytes_per_hour = bites_per_hour / bites_per_byte
    message += 'b) Dupa o ora de achizitie semnalul intregistrat va ocupa {:,.0f} biti sau {:,.0f} bytes.\n'.format(
        bites_per_hour, bytes_per_hour)

    file_name = 'exercitiul_3.txt'
    file = open(os.path.join(output_dir, file_name), 'w')
    file.write(message)
    file.close()


if __name__ == '__main__':
    main()
