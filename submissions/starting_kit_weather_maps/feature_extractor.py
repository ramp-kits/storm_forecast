import pandas as pd
import numpy as np


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_df_new = pd.concat(
            [X_df.get(['instant_t', 'windspeed', 'latitude',
                       'longitude', 'hemisphere', 'Jday_predictor',
                       'initial_max_wind', 'max_wind_change_12h',
                       'dist2land']),
             pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)],
            # 'basin' is not used here ..but it can!
            axis=1)
        X_df_new = X_df_new.fillna(-1)
        XX = X_df_new.values

        # reconstruct also some image data
        # (7 2D image per storm instant: z, u, v, sst, slp, hum, vo700 ):
        grid_l = 11  # size of the image is 11 x 11

        # first, altitude z image (example)
        z_image = np.zeros([grid_l, grid_l, len(X_df)])
        for i in range(grid_l):
            for j in range(grid_l):
                z_image[i, j, :] = X_df['z_' + str(i) + '_' + str(j)].values
        z_image = np.transpose(z_image, [2, 0, 1])

        # Simple ex.1 :
        # compute the mean of the 700hPa-level altitude
        # over the 25x25 deg grid centered on the storm
        z_mean = np.mean(z_image, axis=(1, 2))
        # Simple ex.2 :
        # get the center data of the grid (where is the storm)
        z_center = z_image[:, int(grid_l / 2), int(grid_l / 2)]

        # add columns to the feature matrix
        XX = np.insert(XX, len(XX[0]), z_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), z_center, axis=1)

        # Same with sea surface temperature (sst) 11x11 degree
        sst_image = np.zeros([grid_l, grid_l, len(X_df)])
        for i in range(grid_l):
            for j in range(grid_l):
                sst_image[i, j, :] = \
                    X_df['sst_' + str(i) + '_' + str(j)].values
        sst_image = np.transpose(sst_image, [2, 0, 1])
        sst_mean = np.mean(sst_image, axis=(1, 2))
        sst_center = sst_image[:, int(grid_l / 2), int(grid_l / 2)]
        XX = np.insert(XX, len(XX[0]), sst_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), sst_center, axis=1)

        return XX
