import numpy as np
import os


def extract_histogram(input_filename, output_filename, num_rows, num_columns):

    os.system('rm {}'.format(output_filename))

    hist = np.zeros((num_rows, num_columns))
    input_f = open(input_filename)

    line = input_f.readline()
    line = input_f.readline()
    while line:
        data = line.strip().split('\t')
        column = int(data[0])
        row = int(data[1])
        freq = int(data[3])
        hist[row][column] = freq

        line = input_f.readline()

    np.savetxt(output_filename, hist.astype(int), fmt='%i', delimiter=',')

    input_f.close()


def extract_histograms():
    histogram_dirs = ['128x128']
    # f = open('../data/large_datasets.csv')
    f = open('../data/dataset_filenames/large_aws_filenames.csv')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    for histogram_dir in histogram_dirs:
        data = histogram_dir.split('x')
        num_rows = int(data[0])
        num_columns = int(data[1])
        input_dir = '../data/histograms/raw/{}/large_aws'.format(histogram_dir)
        output_dir = '../data/histograms/{}/large_aws'.format(histogram_dir)
        for filename in filenames:
            extract_histogram('{}/{}'.format(input_dir, filename), '{}/{}'.format(output_dir, filename), num_rows, num_columns)


def extract_histograms_pairs():
    histogram_dirs = ['16x16', '32x32', '64x64']

    f = open('../data/dataset_filenames/small_datasets_non_aligned_filenames.csv')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    for histogram_dir in histogram_dirs:
        data = histogram_dir.split('x')
        num_rows = int(data[0])
        num_columns = int(data[1])

        count = 0
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                if i != j:
                    count += 1
                    input_dir = '../data/histograms_raw/small_datasets_non_aligned_pairs/{}/{}'.format(histogram_dir, count)
                    output_dir = '../data/histograms/small_datasets_non_aligned_pairs/{}/{}'.format(histogram_dir, count)
                    dataset1 = filenames[i]
                    dataset2 = filenames[j]
                    extract_histogram('{}/{}'.format(input_dir, dataset1), '{}/{}'.format(output_dir, dataset1),
                                      num_rows, num_columns)
                    extract_histogram('{}/{}'.format(input_dir, dataset2), '{}/{}'.format(output_dir, dataset2),
                                      num_rows, num_columns)


def shrink_histograms(num_rows, num_columns):
    shrinked_rows = int(num_rows / 2)
    shrinked_columns = int(num_columns / 2)
    input_dir = '../data/histograms/{}x{}/real'.format(num_rows, num_columns)
    output_dir = '../data/histograms/{}x{}/real'.format(shrinked_rows, shrinked_columns)

    f = open('../data/dataset_filenames/real_datasets_intersecting_all_filenames.csv')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    for dataset in filenames:
        hist_input_filename = '{}/{}'.format(input_dir, dataset)
        hist_output_filename = '{}/{}'.format(output_dir, dataset)

        hist_input = np.genfromtxt(hist_input_filename, delimiter=',')
        hist_output = np.zeros((shrinked_rows, shrinked_columns))

        for i in range(hist_output.shape[0]):
            for j in range(hist_output.shape[1]):
                hist_output[i, j] = hist_input[2*i, 2*j] + hist_input[2*i + 1, 2*j] + hist_input[2*i, 2*j + 1] + hist_input[2*i + 1, 2*j + 1]

        np.savetxt(hist_output_filename, hist_output.astype(int), fmt='%i', delimiter=',')


def main():
    print('Compute histogram')
    # extract_histograms()
    # extract_histograms_pairs()
    # shrink_histograms(128, 128)
    # shrink_histograms(64, 64)
    # shrink_histograms(32, 32)


if __name__ == '__main__':
    main()
