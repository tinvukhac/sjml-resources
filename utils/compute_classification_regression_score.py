import numpy as np
import pandas as pd
from sklearn import metrics


def get_test_duration(row, label):
    y_test = row[label]
    if y_test == 1:
        return row['bnlj_duration']
    elif y_test == 2:
        return row['pbsm_duration']
    elif y_test == 3:
        return row['dj_duration']
    else:
        return row['repj_duration']


def main():
    print('Compute classification regression score')
    # test_filenames = ['../data/temp/algorithm_selection_b3_updated_5_31.csv',
    #                   '../data/temp/algorithm_selection_m3_baseline2.csv',
    #                   '../data/temp/algorithm_selection_m3_fs1_v2.csv',
    #                   '../data/temp/algorithm_selection_m3_fs2_v3.csv',
    #                   '../data/temp/algorithm_selection_m3_fs3_v3.csv']
    # test_filenames = ['../data/temp/algorithm_selection_m3_fs4_v3.csv']
    test_filenames = ['../data/temp/test_df.csv']
    join_result_filenames = ['../data/ranked_join_results/join_results_large_aws_x_medium_datasets_ranked.csv',
                            '../data/ranked_join_results/join_results_large_uniform_datasets_ranked.csv',
                            '../data/ranked_join_results/join_results_real_datasets_ranked_with_prefix.csv']

    df = pd.DataFrame()
    for join_result_filename in join_result_filenames:
        df1 = pd.read_csv(join_result_filename, delimiter='\\s*,\\s*', header=0)
        df1 = df1[['dataset1', 'dataset2', 'bnlj_duration', 'pbsm_duration', 'dj_duration', 'repj_duration']]
        df = df.append(df1)

    for test_filename in test_filenames:
        test_df = pd.read_csv(test_filename, delimiter='\\s*,\\s*', header=0)
        test_df = pd.merge(test_df, df, how='left', left_on=['dataset1', 'dataset2'],
                 right_on=['dataset1', 'dataset2'])
        test_df['test_duration'] = test_df.apply(lambda x: get_test_duration(x, 'y_test'), axis=1)
        test_df['pred_duration'] = test_df.apply(lambda x: get_test_duration(x, 'y_pred'), axis=1)
        test_duration = test_df[['test_duration']].astype(float).to_numpy()
        pred_duration = test_df[['pred_duration']].astype(float).to_numpy()
        # print(test_duration)
        # print(pred_duration.shape)
        # test_df.to_csv('test.csv')
        diff = np.divide(abs(pred_duration - test_duration), test_duration)
        regression_score = 1 - np.mean(diff)
        print(np.mean(diff))
        # print(regression_score)
        # mape = metrics.mean_absolute_percentage_error(test_duration, pred_duration)
        # print(mape)


if __name__ == '__main__':
    main()
