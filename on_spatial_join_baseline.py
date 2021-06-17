import pandas as pd


def select_spatial_join(r_avg_x, s_avg_x, r_avg_y, s_avg_y, r_x1, r_y1, r_x2, r_y2, s_x1, s_y1, s_x2, s_y2, r_blocks, s_blocks):
    T = 480

    min_x, min_y, max_x, max_y = min(r_x1, s_x1), min(r_y1, s_y1), max(r_x2, s_x2), max(r_y2, s_y2)
    area = (max_x - min_x) * (max_y - min_y)
    beta_rs = (r_avg_x + s_avg_x) * (r_avg_y + s_avg_y) / area

    join_method = 'PBSM'

    if beta_rs * s_blocks < 1:
        join_method = 'DJ'
    else:
        if r_blocks < T and s_blocks < T:
            join_method = 'PBSM'
        else:
            join_method = "REPJ"

    return join_method


def main():
    print('Baseline: On Spatial Joins in MapReduce')

    df = pd.read_csv('data/train_and_test_all_features_split/test_join_results_combined_v2.csv')
    df['best_algo'] = df.apply(lambda x: select_spatial_join(x['AVG x_x'], x['AVG x_y'], x['AVG y_x'], x['AVG y_y'], x['x1_x'], x['y1_x'], x['x2_x'], x['y2_x'],
                                              x['x1_y'], x['y1_y'], x['x2_y'], x['y2_y'], x['total_blocks_x'], x['total_blocks_y']), axis=1)
    df.to_csv('data/train_and_test_all_features_split/test_join_results_combined_applied_baseline2.csv')


if __name__ == '__main__':
    main()
