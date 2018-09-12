import pandas as pd


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_df.index = range(len(X_df))

        X_df_new = pd.concat(
            [X_df.get(['instant_t', 'windspeed', 'latitude', 'longitude',
                       'hemisphere', 'Jday_predictor', 'initial_max_wind',
                       'max_wind_change_12h', 'dist2land']),
             pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)],
            # 'basin' is not used here ..but it can!
            axis=1)

        # get data from the past of the same storm (if it exists)
        past_winds = []
        for i in range(len(X_df)):
            if i - 1 < 0:
                past_winds.append(X_df['windspeed'][i])
            elif X_df['stormid'][i] == X_df['stormid'][i - 1]:
                past_winds.append(X_df['windspeed'][i - 1])
            else:
                past_winds.append(X_df['windspeed'][i])
        X_df_new = X_df_new.assign(
            past_windspeed=pd.Series(past_winds))

        X_df_new = X_df_new.fillna(-1)
        XX = X_df_new.values
        return XX
