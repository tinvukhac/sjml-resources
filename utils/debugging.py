import matplotlib.pyplot as plt
import pandas as pd


def main():
    print('Debug the data/source code')

    data_files = ['train_join_results_small_x_small.csv',
                  'test_join_results_small_x_small.csv',
                  'train_join_results_large_aws_x_large_aws.csv',
                  'test_join_results_large_aws_x_large_aws.csv',
                  'train_join_results_large_aws_x_large_aws_filtered.csv']

    for data_file in data_files:
        join_df = pd.read_csv('../data/train_and_test_all_features_split/{}'.format(data_file), delimiter='\\s*,\\s*', header=0)
        join_selectivity = join_df['join_selectivity']
        plt.hist(join_selectivity, bins=10)
        plt.show()


if __name__ == '__main__':
    main()
