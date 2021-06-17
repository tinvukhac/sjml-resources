import pickle

import numpy as np
import pandas as pd
from keras.losses import mean_squared_logarithmic_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

import datasets
from model_interface import ModelInterface


class RegressionModel(ModelInterface):
    NORMALIZE = False
    DISTRIBUTION = 'all'
    MATCHED = False
    SCALE = 'all'
    MINUS_ONE = False
    # TARGET = 'join_selectivity'
    TARGET = 'mbr_tests_selectivity'
    # Descriptors
    drop_columns_feature_set1 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'E0_x', 'E2_x', 'total_area_x', 'total_margin_x',
                                 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x', 'total_area_y', 'total_margin_y',
                                 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y', 'intersection_area1', 'intersection_area2', 'jaccard_similarity',
                                 'cardinality_x', 'cardinality_y', 'e0', 'e2']
    # Descriptors + histograms
    drop_columns_feature_set2 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'total_area_x', 'total_margin_x',
                                 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x', 'total_area_y', 'total_margin_y',
                                 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y', 'cardinality_x', 'cardinality_y']
    # Descriptors + histograms + partitioning features
    drop_columns_feature_set3 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'cardinality_x', 'cardinality_y']
    DROP_COLUMNS = drop_columns_feature_set2
    SELECTED_COLUMNS = []

    def __init__(self, model_name):
        self.reg_model = LinearRegression()
        if model_name == 'linear':
            self.reg_model = LinearRegression()
        elif model_name == 'decision_tree':
            self.reg_model = DecisionTreeRegressor(max_depth=8)
        elif model_name == 'random_forest':
            self.reg_model = RandomForestRegressor(max_depth=8, random_state=0)

    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a regression model for spatial join cost estimator, then save the trained model to file
        """

        # Extract train and test data, but only use train data
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features_hadoop(RegressionModel.DISTRIBUTION, RegressionModel.MATCHED, RegressionModel.SCALE, RegressionModel.MINUS_ONE)
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path, RegressionModel.NORMALIZE, RegressionModel.MINUS_ONE, RegressionModel.TARGET)
        X_train, y_train, join_df = datasets.load_data(tabular_path, RegressionModel.TARGET, RegressionModel.DROP_COLUMNS, RegressionModel.SELECTED_COLUMNS)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        # query_val = [X_val.shape[0]]
        #
        # Fit and save the model
        model = self.reg_model.fit(X_train, y_train)
        # model = LGBMRanker()
        # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_group=[query_val], eval_at=[1, 2],
        #           early_stopping_rounds=50)

        pickle.dump(model, open(model_path, 'wb'))

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """

        # Extract train and test data, but only use test data
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features_hadoop(RegressionModel.DISTRIBUTION, RegressionModel.MATCHED, RegressionModel.SCALE, RegressionModel.MINUS_ONE)
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path, RegressionModel.NORMALIZE, RegressionModel.MINUS_ONE, RegressionModel.TARGET)
        X_test, y_test, join_df = datasets.load_data(tabular_path, RegressionModel.TARGET, RegressionModel.DROP_COLUMNS, RegressionModel.SELECTED_COLUMNS)

        # Load the model and use it for prediction
        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X_test)

        # Convert back to 1 - y if need
        if RegressionModel.MINUS_ONE:
            y_test, y_pred = 1 - y_test, 1 - y_pred

        # TODO: delete this dumping action. This is just for debugging
        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df.to_csv('data/temp/test_df.csv')

        # Compute accuracy metrics
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        msle = np.mean(mean_squared_logarithmic_error(y_test, y_pred))

        return mae, mape, mse, msle
