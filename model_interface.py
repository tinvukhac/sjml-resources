class ModelInterface:
    def train(self, tabular_path: str, target: str, model_path: str, model_weights_path=None) -> None:
        """
        Train a regression model for spatial join cost estimator, then save the trained model to file
        """
        pass

    def test(self, tabular_path: str, target: str, model_path: str, model_weights_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_logarithmic_error
        """
        pass
