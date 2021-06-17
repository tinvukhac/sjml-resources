from optparse import OptionParser

from dnn_model import DNNModel
from histogram_dnn_model import HistogramDNNModel
from classification_model import ClassificationModel
from ranking_model import  RankingModel
from regression_model import RegressionModel


def main():
    parser = OptionParser()
    parser.add_option('-m', '--model', type='string', help='Model name: {linear, dnn}')
    parser.add_option('-t', '--tab', type='string', help='Path to the tabular data file(CSV)')
    parser.add_option('-g', '--hist', type='string', help='Path to the histograms of input datasets')
    parser.add_option('-r', '--result', type='string', help='Path to the join result (CSV)')
    parser.add_option('-p', '--path', type='string', help='Path to the model to be saved')
    parser.add_option('-w', '--weights', type='string', help='Path to the model weights to be saved')
    parser.add_option('--train', action="store_true", dest="train", default=True)
    parser.add_option('--no-train', action="store_false", dest="train")

    (options, args) = parser.parse_args()
    options_dict = vars(options)

    model_names = ['linear', 'decision_tree', 'random_forest', 'dnn', 'hist_dnn', 'clf_decision_tree', 'clf_random_forest', 'rnk_random_forest']

    try:
        model_name = options_dict['model']
        if model_name not in model_names:
            print('Available model are {}'.format(', '.join(model_names)))
            return
        else:
            if model_name in ['linear', 'decision_tree', 'random_forest']:
                model = RegressionModel(model_name)
            elif model_name == 'dnn':
                model = DNNModel()
            elif model_name == 'hist_dnn':
                model = HistogramDNNModel()
            elif model_name in ['clf_decision_tree', 'clf_random_forest']:
                model = ClassificationModel(model_name)
            elif model_name in ['rnk_random_forest']:
                model = RankingModel(model_name)

        tabular_path = options_dict['tab']
        histogram_path = options_dict['hist']
        join_result_path = options_dict['result']
        model_path = options_dict['path']
        model_weights_path = options_dict['weights']
        is_train = options_dict['train']

        if is_train:
            model.train(tabular_path, join_result_path, model_path, model_weights_path, histogram_path)
        else:
            mae, mape, mse, msle = model.test(tabular_path, join_result_path, model_path, model_weights_path, histogram_path)
            if model_name in ['clf_decision_tree', 'clf_random_forest']:
                exit(1)
            print('mae: {}\nmape: {}\nmse: {}\nmlse: {}'.format(mae, mape, mse, msle))
            print('{}\t{}\t{}\t{}'.format(mae, mape, mse, msle))

    except RuntimeError:
        print('Please check your arguments')


if __name__ == "__main__":
    main()
