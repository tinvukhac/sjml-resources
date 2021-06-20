from optparse import OptionParser

from classification_model import ClassificationModel
from regression_model import RegressionModel


def main():
    parser = OptionParser()
    parser.add_option('-m', '--model', type='string', help='Model name: {linear, dnn}')
    parser.add_option('-t', '--tab', type='string', help='Path to the tabular data file(CSV)')
    parser.add_option('-l', '--target', type='string', help='Target (performance metric) of the estimation model')
    parser.add_option('-p', '--path', type='string', help='Path to the model to be saved')
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
            elif model_name in ['clf_decision_tree', 'clf_random_forest']:
                model = ClassificationModel(model_name)

        tabular_path = options_dict['tab']
        target = options_dict['target']
        model_path = options_dict['path']
        is_train = options_dict['train']

        if is_train:
            model.train(tabular_path, target, model_path)
        else:
            mae, mape, mse, msle = model.test(tabular_path, target, model_path)
            if model_name in ['clf_decision_tree', 'clf_random_forest']:
                exit(1)
            print('mae: {}\nmape: {}\nmse: {}\nmlse: {}'.format(mae, mape, mse, msle))
            print('{}\t{}\t{}\t{}'.format(mae, mape, mse, msle))

    except RuntimeError:
        print('Please check your arguments')


if __name__ == "__main__":
    main()
