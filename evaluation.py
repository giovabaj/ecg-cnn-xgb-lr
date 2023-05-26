import numpy as np
from sklearn.metrics import roc_auc_score

from rpy2.robjects import r
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()


def ici(labels, predictions):
    """Function to compute the integrated calibration index for a binary outcome.
    Smooth calibration curves based on local polynomial regression is produced regressing the binary outcome with the
    predicted risk. Then the ICI is the weighted difference between smoothed observed proportions and predicted probs.

    Reference: Austin and Steyerberg (2019). https://doi.org/10.1002/sim.8281

    Input
     - labels: observed binary outcomes
     - predictions: predicted probabilities
    Output
     - ici: integrated calibration index
    """
    # Converting arrays to R
    y, p = robjects.FloatVector(labels), robjects.FloatVector(predictions)
    df = robjects.DataFrame({"P": p, "Y": y})
    # Fitting the model and making predictions
    loess_fit = r.loess("Y ~ P", data=df)
    p_calibrate = np.array(r.predict(loess_fit, newdata=p))
    # Computing ICI
    ici_ = np.mean(np.abs(p_calibrate-predictions))
    return ici_


class MetricsCV:
    """Class to keep track of the metrics in all the different cross-validation trainings.
    """
    def __init__(self, path_results, set_name):
        self.auc = []
        self.ici = []
        self.path_results = path_results
        self.set_name = set_name

    def compute_metrics(self, labels, predictions):
        self.auc.append(roc_auc_score(labels, predictions))
        self.ici.append(ici(labels, predictions))

    def save_results(self):
        np.save(self.path_results + 'auc_' + self.set_name, self.auc)
        np.save(self.path_results + 'ici_' + self.set_name, self.ici)
