Ramp kit storm forecast
=======================


[![Build Status](https://travis-ci.org/sophiegif/ramp_kit_storm_forecast.svg?branch=master)][travis]

_Authors: Sophie Giffard-Roisin_

The goal is to predict the hurrican evolution (24h forecast) using collected data from all past hurricanes (since 1979)

## Set up

1. clone this repository
  ```
  git clone https://github.com/ramp-kits/storm_forecast.git
  cd storm_forecast
  ```

2. install the dependancies
  - with [conda](https://conda.io/miniconda.html)
  ```
  conda install -y -c conda conda-env     # First install conda-env
  conda env create                        # Use environment.yml to create the 'storm_forecast' env
  source activate storm_forecast       # Activates the virtual env
  ```
  - without `conda` (best to use a **virtual environment**)
  ```
  python -m pip install -r requirements.txt
  ```

3. download the data
  ```
  python download_data.py        # quick-test data for testing ~270Mb
  ```

4. get started with the storm_forecast_starting_kit.ipynb

## New submissions

1. create a new submission "<new_sub>" by building on the existing ones
  ```
  cp -r submissions/starting_kit submissions/<new_sub>
  ```
2. modify the `*.py` files in  `submissions/<new_sub>` with your favorite editor

3. test the submission with
  ```
  ramp_test_submission --quick-test --submission <new_sub>
  ```
4. if the job complete, you can submit the code in the sandbox of [ramp.studio][ramp]



## License

BSD license : see [LICENSE file](LICENSE)


## Credits

This package was created with [Cookiecutter][cookie] and the [`ramp-kits/cookiecutter-ramp-kit`][kit] project template
issued by the [Paris-Saclay Center for Data Science][cds].

[travis]: https://travis-ci.org/sophiegif/ramp_kit_storm_forecast
[ramp]: https://ramp.studio/events/storm_forecast
[cookie]: https://github.com/audreyr/cookiecutter
[kit]: https://github.com/ramp-kits/cookiecutter-ramp-kit
[cds]: https://www.datascience-paris-saclay.fr/
