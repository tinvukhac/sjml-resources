from itertools import combinations
import os
from regression_model import RegressionModel


def main():
    print('Train and test in batch')

    train_results_f = open('data/temp/train_results.csv', 'w')

    distributions = ['uniform', 'diagonal', 'gauss', 'parcel', 'bit']

    test_path = 'data/train_and_test_all_features_split/test_join_results_combined_data.csv'

    for r in range(1, len(distributions)):
        print(r)
        groups = combinations(distributions, r)
        for g in groups:
            name = '_'.join(g)
            output_name = '{}distribution.{}'.format(r, name)
            train_path = 'data/train_and_test_all_features_split/train_join_results_combined_data.{}.csv'.format(output_name)
            print(train_path)
            os.system('python main.py --model random_forest --tab {} --hist data/histograms/ --result data/join_results/train/join_results_small_x_small_uniform.csv --path trained_models/model_uniform.h5 --weights trained_models/model_weights_uniform.h5 --train'.format(train_path))
            # os.system('python main.py --model random_forest --tab data/train_and_test_all_features_split/test_join_results_combined_data.csv --hist data/histograms/ --result data/join_results/train/join_results_small_x_small_uniform.csv --path trained_models/model_uniform.h5 --weights trained_models/model_weights_uniform.h5 --no-train')
            model = RegressionModel('random_forest')
            mae, mape, mse, msle = model.test('data/train_and_test_all_features_split/test_join_results_combined_data.csv',
                                                        '', 'trained_models/model_uniform.h5', 'trained_models/model_weights_uniform.h5', '')
            train_results_f.writelines('{},{},{},{},{},{}\n'.format(r, name, mae, mape, mse, msle))

    train_results_f.close()


if __name__ == '__main__':
    main()
