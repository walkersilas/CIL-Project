import numpy as np
import surprise

# Hyper Parameters used for the Model
hyper_parameters = {}


class SlopeOne():
    def __init__(self, train_data, test_data, test_ids, args, config, logger):
        self.args = args

        # Configuration used for execution
        self.config = config

        self.logger = logger

        self.train_data = train_data
        self.test_data = test_data
        self.test_ids = test_ids.to_numpy()

        self.algo = surprise.SlopeOne()

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
