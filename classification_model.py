import pickle

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import time
import numpy as np

import datasets
from model_interface import ModelInterface

import matplotlib.pyplot as plt


class ClassificationModel(ModelInterface):
    NORMALIZE = False
    DISTRIBUTION = 'all'
    MATCHED = False
    SCALE = 'all'
    MINUS_ONE = False
    # TARGET = 'join_selectivity'
    # TARGET = 'mbr_tests_selectivity'
    TARGET = 'best_algorithm'
    # Descriptors
    drop_columns_feature_set1 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'E0_x', 'E2_x', 'E0_y', 'E2_y', 'total_area_x', 'total_margin_x',
                                 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x', 'total_area_y', 'total_margin_y',
                                 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y', 'intersection_area1', 'intersection_area2', 'jaccard_similarity',
                                 'cardinality_x', 'cardinality_y', 'e0', 'e2', 'cardinality_x',	'AVG area_x', 'AVG x_x', 'AVG y_x',
                                 'cardinality_y',	'AVG area_y', 'AVG x_y', 'AVG y_y']
    feature_set1 = ['cardinality_x', 'AVG area_x', 'AVG x_x', 'AVG y_x',
                    'cardinality_y', 'AVG area_y', 'AVG x_y', 'AVG y_y']
    # Descriptors + histograms
    drop_columns_feature_set2 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'total_area_x', 'total_margin_x',
                                 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x', 'total_area_y', 'total_margin_y',
                                 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y', 'cardinality_x', 'cardinality_y']
    feature_set2 = ['cardinality_x', 'AVG area_x', 'AVG x_x', 'AVG y_x', 'E0_x', 'E2_x',
                    'cardinality_y', 'AVG area_y', 'AVG x_y', 'AVG y_y', 'E0_y', 'E2_y',
                    'intersection_area1', 'intersection_area2', 'jaccard_similarity', 'e0', 'e2']
    # Descriptors + histograms + partitioning features
    drop_columns_feature_set3 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'cardinality_x', 'cardinality_y']
    feature_set3 = ['cardinality_x', 'AVG area_x', 'AVG x_x', 'AVG y_x', 'E0_x', 'E2_x', 'block_size_x', 'total_area_x', 'total_margin_x', 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x',
                    'cardinality_y', 'AVG area_y', 'AVG x_y', 'AVG y_y', 'E0_y', 'E2_y', 'block_size_y', 'total_area_y', 'total_margin_y', 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y',
                    'intersection_area1', 'intersection_area2', 'jaccard_similarity', 'e0', 'e2']
    # Join cardinality + MBRs of 4 algorithms
    feature_set4 = ['join_cardinality', 'mbr_bnlj', 'mbr_pbsm', 'mbr_dj', 'mbr_repj']
    feature_set5 = ['join_cardinality', 'mbr_bnlj', 'mbr_pbsm', 'mbr_dj', 'mbr_repj', 'block_size_x', 'block_size_y']
    feature_set6 = ['cardinality_x', 'AVG area_x', 'AVG x_x', 'AVG y_x', 'E0_x', 'E2_x', 'block_size_x', 'total_area_x', 'total_margin_x', 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x',
                    'cardinality_y', 'AVG area_y', 'AVG x_y', 'AVG y_y', 'E0_y', 'E2_y', 'block_size_y', 'total_area_y', 'total_margin_y', 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y',
                    'intersection_area1', 'intersection_area2', 'jaccard_similarity', 'e0', 'e2', 'join_cardinality', 'mbr_bnlj', 'mbr_pbsm', 'mbr_dj', 'mbr_repj']
    feature_set7 = ['join_cardinality', 'mbr_bnlj', 'mbr_pbsm', 'mbr_dj', 'mbr_repj', 'block_size_x', 'block_size_y', 'intersection_area1', 'intersection_area2', 'jaccard_similarity']

    DROP_COLUMNS = []
    SELECTED_COLUMNS = feature_set6

    def __init__(self, model_name):
        self.clf_model = DecisionTreeClassifier()
        if model_name == 'clf_decision_tree':
            self.clf_model = DecisionTreeClassifier(max_depth=8)
        elif model_name == 'clf_random_forest':
            self.clf_model = RandomForestClassifier(max_depth=8, random_state=0)

    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a classification model for spatial join cost estimator, then save the trained model to file
        """

        # Extract train and test data, but only use train data
        X_train, y_train, join_df = datasets.load_data(tabular_path, ClassificationModel.TARGET, ClassificationModel.DROP_COLUMNS, ClassificationModel.SELECTED_COLUMNS)

        # Fit and save the model
        model = self.clf_model.fit(X_train, y_train)
        pickle.dump(model, open(model_path, 'wb'))

        # Feature importances
        importances = model.feature_importances_

        output_f = open('data/temp/feature_importances.csv', 'w')
        output_f.writelines('feature_name,importance_score\n')

        for fname, fscore in zip(ClassificationModel.SELECTED_COLUMNS, importances):
            print('{},{}'.format(fname, fscore))
            output_f.writelines('{},{}\n'.format(fname, fscore))

        output_f.close()

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """

        # Extract train and test data, but only use test data
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features_hadoop(RegressionModel.DISTRIBUTION, RegressionModel.MATCHED, RegressionModel.SCALE, RegressionModel.MINUS_ONE)
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path, RegressionModel.NORMALIZE, RegressionModel.MINUS_ONE, RegressionModel.TARGET)
        X_test, y_test, join_df = datasets.load_data(tabular_path, ClassificationModel.TARGET, ClassificationModel.DROP_COLUMNS, ClassificationModel.SELECTED_COLUMNS)

        # Load the model and use it for prediction
        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X_test)

        # TODO: delete this dumping action. This is just for debugging
        test_df = pd.DataFrame()
        test_df['dataset1'] = join_df['dataset1']
        test_df['dataset2'] = join_df['dataset2']
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df.to_csv('data/temp/test_df.csv', index=None)

        # Compute accuracy metrics
        acc = metrics.accuracy_score(y_test, y_pred)
        print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

        # Plot non-normalized confusion matrix
        titles_options = [("figures/confusion_matrix_without_normalization.png", None),
                          ("figures/confusion_matrix_with_normalization.png", 'true')]
        class_names = ['BNLJ', 'PBSM', 'DJ', 'RepJ']
        for title, normalize in titles_options:
            plt.rcParams.update({'font.size': 14})
            disp = plot_confusion_matrix(loaded_model, X_test, y_test,
                                         display_labels=class_names,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title("")

            print(title)
            print(disp.confusion_matrix)
            plt.xlabel('Predicted algorithm', fontsize=16)
            plt.ylabel('Actual best algorithm', fontsize=16)
            plt.savefig(title)

        return acc, acc, acc, acc
