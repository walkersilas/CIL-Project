import numpy as np
import surprise

# Hyper Parameters used for the Model
hyper_parameters = {
    'k': 99,
    'min_k': 11,
    'name': "pearson_baseline",
    'user_based': False,
    'method': "als"
}


class KNNBaseline():
    def __init__(self, train_data, test_data, test_ids, args, config, logger):
        self.args = args

        # Configuration used for execution
        self.config = config

        self.logger = logger

        self.train_data = train_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        self.k = config['k']
        self.min_k = config['min_k']
        self.name = config['name']
        self.user_based = config['user_based']
        self.method = config["method"]
        self.bsl_options = {"method": self.method}
        self.sim_options = {"name": self.name, "user_based": self.user_based}

        self.algo = surprise.KNNBaseline(k=self.k,
                                         min_k=self.min_k,
                                         sim_options=self.sim_options,
                                         bsl_options=self.bsl_options)

    def fit(self):
        self.algo.fit(self.train_data)

    def test(self):
        predictions = []
        for user, movie in self.test_data:
            prediction = self.algo.predict(user.item(), movie.item()).est
            predictions.append(prediction)
        prediction_output = np.stack((self.test_ids, predictions), axis=1)
        self.logger.experiment.log_table(filename="predictions.csv",
                                         tabular_data=prediction_output,
                                         headers=["Id", "Prediction"])
