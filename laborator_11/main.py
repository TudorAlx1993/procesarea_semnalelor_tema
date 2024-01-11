import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():
    output_dir = './outputs'

    mean = 5.0
    variance = 0.5
    no_samples = 10 ** 5
    ex_1_unidimensional(mean, variance, no_samples, os.path.join(output_dir, 'ex_1_unidimensional'))

    mean_vector = np.array([2.0,2.5])
    covariance_matrix = np.array([[0.5, 0.35], [0.35, 0.25]])
    ex_1_bidimensional(mean_vector, covariance_matrix, no_samples, os.path.join(output_dir, 'ex_1_bidimensional'))


def ex_1_bidimensional(mean_vector, covariance_matrix, no_samples, output_dir):
    if not (isinstance(mean_vector, np.ndarray) and isinstance(covariance_matrix, np.ndarray)):
        raise ValueError('parameters mean_vector and covariance_matrix should be numpy arrays!')

    if not mean_vector.ndim == 1:
        raise ValueError('parameter mean_vector should be a 1D numpy array!')

    if not (covariance_matrix.ndim == 2 and np.allclose(covariance_matrix, covariance_matrix.T)):
        raise ValueError('parameter covariance_matrix should be a 2D symmetrical numpy array!')

    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    u_matrix = eigen_vectors
    lambda_matrix = np.diag(eigen_values)

    samples = np.random.multivariate_normal(mean=np.zeros_like(mean_vector),
                                            cov=np.identity(covariance_matrix.ndim),
                                            size=no_samples)
    values = np.zeros_like(samples)

    values = np.matmul(u_matrix, (np.sqrt(np.diag(lambda_matrix)) * samples).transpose()).transpose() + mean_vector

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data = {'samples': samples,
            'values': values,
            'mean_vector': mean_vector,
            'covariance_matrix': covariance_matrix}

    file = open(os.path.join(output_dir, 'simulated_data.pkl'), 'wb')
    pickle.dump(data, file)
    file.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(values[:, 0], values[:, 1], color='blue')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title('Simulated 2D Gaussian Process')
    fig.savefig(os.path.join(output_dir, 'plotted_data.png'))
    fig.savefig(os.path.join(output_dir, 'plotted_data.pdf'), format='pdf')


def ex_1_unidimensional(mean, variance, no_samples, output_dir):
    if not isinstance(mean, float):
        assert ValueError('parameter mean should be a float!')

    if not (isinstance(variance, float) and variance > 0.0):
        assert ValueError('parameter standard_deviation should be of float type and strictly positive!')

    samples = np.random.normal(loc=0.0, scale=1.0, size=no_samples)
    values = np.sqrt(variance) * samples + mean

    assert np.allclose(np.mean(values), mean, rtol=0.01, atol=0.01)
    assert np.allclose(np.std(values) ** 2, variance, rtol=0.01, atol=0.01)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data = {'samples': samples,
            'values': values,
            'mean': mean,
            'standard_deviation': variance}

    file = open(os.path.join(output_dir, 'simulated_data.pkl'), 'wb')
    pickle.dump(data, file)
    file.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(values, density=True, color='blue', bins=100)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title('Unidimensional Gaussian Process with mean={:.4f} and standard deviation={:.4f}'.format(mean,
                                                                                                         variance))
    fig.savefig(os.path.join(output_dir, 'plotted_data.png'))
    fig.savefig(os.path.join(output_dir, 'plotted_data.pdf'), format='pdf')


if __name__ == '__main__':
    main()
