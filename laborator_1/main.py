import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # plot-urile si raspunsurile vor fi salvate in directorul outputs
    output_dir = './outputs'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # exercise_1(output_dir)
    # exercise_2(output_dir)
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
    pass


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
