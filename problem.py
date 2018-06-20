import pandas as pd
import numpy as np
import math
import copy
import rampwf
from rampwf.workflows.regressor import Regressor
from rampwf.utils.importing import import_file
from rampwf.score_types import RMSE
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
pd.options.mode.chained_assignment = None

problem_title = 'Storm forecast'
_forecast_h = 24

# --------------------------------------
# 1) The object implementing the workflow
# --------------------------------------


class StormForecastFeatureExtractor(object):
    def __init__(self, check_indexs,
                 workflow_element_names=['feature_extractor']):
        self.element_names = workflow_element_names
        self.check_indexs = list(check_indexs)

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        feature_extractor = import_file(module_path, self.element_names[0])

        sf_fe = feature_extractor.FeatureExtractor()
        sf_fe.fit(X_df.iloc[train_is], y_array[train_is])
        return sf_fe

    def test_submission(self, trained_model, X_df):
        sf_fe = trained_model
        X_df.index = range(len(X_df))
        X_test_array = sf_fe.transform(X_df)

        # Check if feature extractor look ahead of time on some storms.
        # For that, use the check_indexs elements (consitisting of one time
        # of one storm) and look if it uses information from a future timestep
        # of the same storm. The pandas feature matrix X_df has 2 important
        # columns X_df['stormid'] and X_df['instant_t']
        # storing the stormid and the time instant of every data point.

        future_indexs = []
        for check_index in self.check_indexs:
            if check_index > len(X_df):
                self.check_indexs.remove(check_index)
                continue
            curr_stormid = X_df['stormid'][check_index]
            curr_tstep = X_df['instant_t'][check_index]
            # get ids of the matrix X_df where are stored the features of
            # the following timesteps of the same stormid
            # -should be just after-
            f_indexs = []
            for index in range(check_index + 1, len(X_df)):
                if curr_stormid == X_df['stormid'][index]:
                    if curr_tstep > X_df['instant_t'][index]:
                        print('test_submission: Something is wrong...'
                              'the time instant should be '
                              'larger than current!')
                    f_indexs.append(index)
                else:
                    break
            # verify if check_index not in future_indexs
            # (one check max. per storm)
            if set(self.check_indexs) & set(f_indexs):  # look at intersection
                self.check_indexs.remove(check_index)
                continue
            future_indexs.extend(f_indexs)

        # permute data from the future_indexs for every column :
        #  it can be applied to any kind of data
        data_var_names = list(X_df.keys())
        keep_data = {}
        for data_var_name in data_var_names:
            keep_data[data_var_name] = \
                copy.deepcopy(X_df[data_var_name][future_indexs])
            X_df.loc[future_indexs, data_var_name] = \
                np.random.permutation(X_df[data_var_name][future_indexs])
        # calling transform on changed future
        X_check_array = sf_fe.transform(X_df)
        # set X_df normal again
        for data_var_name in data_var_names:
            X_df.loc[future_indexs, data_var_name] = keep_data[data_var_name]

        X_neq = np.not_equal(
            X_test_array[self.check_indexs], X_check_array[self.check_indexs])
        x_neq = np.any(X_neq, axis=1)
        x_neq_nonzero = x_neq.nonzero()
        # The final features should not have changed
        # between the first transform()
        # and the transform() where the future has been modified.
        # if not, than it means an illegal lookahead has happened.
        if len(x_neq_nonzero[0]) > 0:
            message = 'The feature extractor looks into the future ' \
                      'timesteps of the same storm!'
            raise AssertionError(message)

        return X_test_array


class StormForecastFeatureExtractorRegressor(object):
    def __init__(self, check_indexs,
                 workflow_element_names=['feature_extractor', 'regressor']):
        self.element_names = workflow_element_names
        self.sf_feature_extractor_workflow = StormForecastFeatureExtractor(
            check_indexs, [self.element_names[0]])
        self.regressor_workflow = Regressor([self.element_names[1]])

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        sf_fe = self.sf_feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_is)
        X_train_array = self.sf_feature_extractor_workflow.test_submission(
            sf_fe, X_df.iloc[train_is])
        reg = self.regressor_workflow.train_submission(
            module_path, X_train_array, y_array[train_is])
        return sf_fe, reg

    def test_submission(self, trained_model, X_df):
        fe, reg = trained_model
        X_test_array = self.sf_feature_extractor_workflow.test_submission(
            fe, X_df)
        y_pred = self.regressor_workflow.test_submission(reg, X_test_array)
        return y_pred


workflow = StormForecastFeatureExtractorRegressor(
    check_indexs=np.random.randint(0, 500, 10))

# -------------------------------------------------------------------
# 2) The prediction type (class) to create wrapper objects for `y_pred`,
# -------------------------------------------------------------------

Predictions = rampwf.prediction_types.make_regression()


# ----------------------------------------------------------------
# 3) The list of metrics to test the predictions against
# ----------------------------------------------------------------

# The 'hurricanes' metrics are only taking into account the time instants
# where the storm is a 'hurricane' and not a depression.
# the difference is the maximal sustained windspeed,
# that is >64 knots for hurricanes.
#  This metric is useful because meteorologists
# compare their predictions on only
# 'hurricane' instants. See https://www.nhc.noaa.gov/verification/verify2.shtml
class MAE(rampwf.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='mae', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.average(np.abs(y_pred - y_true))


class MAE_hurricanes(rampwf.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='mae_hurr', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_true = np.transpose(y_true)
        # the definition of the 1rst hurricane category is
        # a max. wind > 64 knots.
        flag_ishurricane = y_true > 64

        y_true_hurr = y_true[flag_ishurricane]
        y_pred_hurr = y_pred[flag_ishurricane]
        return np.average(np.abs(y_pred_hurr - y_true_hurr))


class RelativeMAE_hurricanes(rampwf.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rel_mae_hurr', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        # the definition of the 1rst hurricane category is
        # a max. wind > 64 knots.
        flag_ishurricane = y_true > 64
        y_true_hurr = y_true[flag_ishurricane]
        y_pred_hurr = y_pred[flag_ishurricane]
        return np.average(np.abs(y_pred_hurr - y_true_hurr) / y_true_hurr)


score_types = [RMSE(name='rmse', precision=3),
               MAE(name='mae', precision=3),
               MAE_hurricanes(name='mae_hurr', precision=3),
               RelativeMAE_hurricanes(name='rel_mae_hurr', precision=3)]


# --------------------------------------------------------------------------
# 4) The cross-validation scheme in the form of a function that returns a list
#    of array indices for the training AND testing data
# --------------------------------------------------------------------------
def get_cv(X, y):
    group = np.array(X['stormid'])
    X, y, group = shuffle(X, y, group, random_state=3)
    gkf = GroupKFold(n_splits=5).split(X, y, group)
    return gkf


# ------------------------------------------------------------
# 5) The method that can read both the training and testing data
# ------------------------------------------------------------
#
#   def _read_data(path, dataset):
#       import pandas
#       data = pandas.read_csv(
#           os.path.join(path, 'data', '{}.csv'.format(dataset)))
#       y = data[<target_column_name>].values
#       X = data.drop([<target_column_name>], axis=1).values
#       return X, y
#
# For heavy datesets, a smaller version of the data should be provided
# as `_mini` files e.g. for `train.csv` file, provide a `train_mini.csv`.
# Therefor the `_read_data` method should implement a way to read these
# smaller files using as trigger the environment variable `RAMP_TEST_MODE`.
#
#   def _read_data(path, dataset):
#       import os
#       if os.getenv('RAMP_TEST_MODE', 0):
#           suffix = '_mini'
#       else:
#           suffit = ''
#       data = pandas.read_csv(
#           os.path.join(path, 'data', '{}{}.csv'.format(dataset, suffix)))
#       ...
#       return X, y

def _read_data(path, dataset):
    try:
        Data = pd.read_csv(path + '/data/' + dataset + '.csv')
    except IOError:
        raise IOError("'data/{0}.csv' is not found. "
                      "Ensure you ran 'python download_data.py' to "
                      "obtain the train/test data".format(dataset))

    y = np.empty((len(Data)))
    y[:] = np.nan
    # only one instant every 6 h,
    # so the forecast window is 'windowt' timesteps ahead
    windowt = _forecast_h / 6
    for i in range(len(Data)):
        if i + windowt >= len(Data):
            continue
        if Data['instant_t'][i + windowt] - Data['instant_t'][i] == windowt:
            y[i] = Data['windspeed'][i + windowt]
    X = Data
    i_toerase = []
    for i, yi in enumerate(y):
        if math.isnan(yi):
            i_toerase.append(i)
    X = X.drop(X.index[i_toerase])
    X.index = range(len(X))
    y = np.delete(y, i_toerase, axis=0)
    # y[y == 0] = 10
    return X, y


def get_test_data(path='.'):
    return _read_data(path, 'test')


def get_train_data(path='.'):
    return _read_data(path, 'train')
