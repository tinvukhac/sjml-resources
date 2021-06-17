import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    print('Plot histograms')
    df = pd.read_csv('../data/train_and_test_all_features/join_results_small_x_small_uniform.csv', delimiter='\\s*,\\s*', header=0)

    join_selectivity = df['join_selectivity']
    mbr_tests_selectivity = df['mbr_tests_selectivity']

    plt.hist(join_selectivity, bins=20)
    plt.xlabel('Join selectivity')
    plt.ylabel('Frequency')
    plt.savefig('../figures/histogram_join_selectivity.png')

    # plt.hist(mbr_tests_selectivity, bins=20)
    # plt.xlabel('MBR tests selectivity')
    # plt.ylabel('Frequency')
    # plt.savefig('../figures/histogram_mbr_tests_selectivity.png')


if __name__ == '__main__':
    main()
