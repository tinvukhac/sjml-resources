import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

def compute_coff1():
    def main():
        print('Compute coff values of join selectivity, MBR test and time')

        normalized = True
        cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
        df = pd.read_csv('../data/join_results/train/join_results_large_datasets.csv', header=None, names=cols)

        X = df[['result_size', 'mbr_tests']].values
        if normalized:
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(X)
            X = pd.DataFrame(x_scaled)
        y = df[['duration']].values
        reg = LinearRegression().fit(X, y)
        print(reg.score(X, y))
        print(reg.coef_)

        prediction_df = pd.DataFrame()
        y_pred = reg.predict(X)
        prediction_df['estimated_time'] = y_pred.flatten()
        prediction_df['actual_time'] = y
        prediction_df.to_csv('../data/join_results/join_running_time_prediction.csv', index=False)

        # Create graphic data
        f = open('../data/join_results/join_running_time_prediction.csv')

        output_f = open('../data/join_results/prediction/join_results_large_datasets.txt', 'w')

        line = f.readline()
        line = f.readline()

        while line:
            data = line.strip().split(',')
            output_f.writelines('{} {} '.format(data[0], data[1]))

            line = f.readline()

        output_f.close()
        f.close()


def compute_coff2():
    print('Compute coff values of join selectivity, MBR test and time')

    algorithms = ['bnlj', 'pbsm', 'dj', 'repj']

    for algorithm in algorithms:
        normalized = True
        # cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
        # df = pd.read_csv(
        #     'data/join_results/prediction/join_results_real_datasets_zcurve_12_2.all_algo.csv', header=0)
        # df = pd.read_csv('data/join_results/prediction/join_results_large_and_non_indexed_medium_datasets.12_2.all_algo.csv', header=0)
        df = pd.read_csv(
            '../data/join_results/prediction/datasets_new_join_cost_model.csv',
            header=0)
        # print(df)
        # df = df[~df['{}_duration'.format(algorithm)].isnull()]
        df = df.dropna(subset=['{}_duration'.format(algorithm), '{}_result_size'.format(algorithm), '{}_mbr_tests'.format(algorithm)])
        df = df[df['{}_duration'.format(algorithm)] != "error"]
        df = df[df['{}_result_size'.format(algorithm)] != "error"]
        df = df[df['{}_mbr_tests'.format(algorithm)] != "error"]
        df = df[df['{}_result_size'.format(algorithm)] != "-1"]
        # df = df[df['{}_duration'.format(algorithm)] < 1000]
        df = df[pd.to_numeric(df['{}_duration'.format(algorithm)], errors='coerce') < 2000]
        # print(len(df))

        X = df[['{}_result_size'.format(algorithm), '{}_mbr_tests'.format(algorithm)]]
        # X.to_csv('test.csv')
        # print(X)
        if normalized:
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(X)
            X = pd.DataFrame(x_scaled)
        y = df[['{}_duration'.format(algorithm)]].values
        reg = LinearRegression().fit(X, y)
        print('{}\t{}'.format(algorithm, reg.score(X, y)))
        # print(reg.coef_)

        prediction_df = pd.DataFrame()
        y_pred = reg.predict(X)
        prediction_df['estimated_time'] = y_pred.flatten()
        prediction_df['actual_time'] = y
        prediction_df.to_csv('../data/join_results/prediction/join_running_time_prediction_{}.csv'.format(algorithm), index=False)

        # Create graphic data
        f = open('../data/join_results/prediction/join_running_time_prediction_{}.csv'.format(algorithm))

        output_f = open('../data/join_results/prediction/join_results_large_datasets_{}.txt'.format(algorithm), 'w')

        line = f.readline()
        line = f.readline()

        while line:
            data = line.strip().split(',')
            output_f.writelines('{} {} '.format(data[0], data[1]))

            line = f.readline()

        output_f.close()
        f.close()


def main():
    compute_coff2()


if __name__ == '__main__':
    main()

