import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    output_dir = './outputs_during_lab_time'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # in timpul acestui laborator am refacut si exercitiul cu spectograma de la tema pentru astazi

    exercise_1('./inputs/train.csv', output_dir)


def exercise_1(file_name, output_dir):
    # citim si formatarea datelor
    data = pd.read_csv(file_name)
    data.columns = map(str.upper, data.columns)
    data.DATETIME = pd.to_datetime(data.DATETIME, format='%d-%m-%Y %H:%M')

    # raspunsurile le scriu in fisier
    file_content = 'Exercitiul 1\n\n'

    # punctul a)
    hours_between_samples = np.unique(np.diff(data.DATETIME.values).astype('timedelta64[h]'))
    # ma asigur ca nu exista pauze de timp intre masuratori
    assert len(hours_between_samples) == 1
    hours_between_samples = hours_between_samples[0].astype(int)
    file_content += 'Punctul a):\n' + \
                    '\t* inregistrarile din fisierul de input apar la intervale de {} ore\n'.format(
                        hours_between_samples)

    output_file = open(os.path.join(output_dir, 'exercise_1.txt'), 'w')
    output_file.write(file_content)
    output_file.close()


if __name__ == '__main__':
    main()
