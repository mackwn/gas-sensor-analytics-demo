<h1>gas-sensor-analytics-demo</h1>
==============================

<p>Anaylyze data from a chemical sensor array responses to different gases at different concentrations. Result is a model to predict identity of the gas and the concentration. Data from UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations#

The application downloads the data, processes the data, and trains a robust model for identifying the gas and concentration based on the sensor response. The sample no. is included as predictor to account for sensor drift. 

The justifcation for the preprocessing steps, feature selection, and model selection is contained in the notebooks (see section below for brief summaries).  

Getting started
===============

-Create the data set by running python -m src.data.make_dataset (run the file manually to change the default file names).
-Review the data in notebooks 1-5
-Train the models by running python -m src.models.train_models. Edit train_models as ncessary based on the model building notebooks.  
-Review the performance of the model in notebook 6

Citations:
A Vergara, S Vembu, T Ayhan, M Ryan, M Homer, R Huerta. "Chemical gas sensor drift compensation using classifier ensembles." Sensors and Actuators B: Chemical 166 (2012): 320-329.

I Rodriguez-Lujan, J Fonollosa, A Vergara, M Homer, R Huerta. "On the calibration of sensor arrays for pattern recognition using the minimal number of experiments." Chemometrics and Intelligent Laboratory Systems 130 (2014): 123-134.</p>

<h2>Project Organization</h2>
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                          and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<h2>Notebooks</h2>
<ul>
<li>1-initial-exploration: Make the dataframe from raw .dat files. Histograms of input features.    Scaling data. </li>
<li>2-build-pipeline: background work for the preprocessing pipeline
<li>3-pca-visualization: Data visualization with using PCA to conceptually view trends in the data </li>
<li>4-explore-model-building-classification: Evaluate classifcation models for predicting the type of gas. </li>
<li>5-explore-model-building-classification: Evaluate regression models for predicting the concentration of a gas. </li>
<li>6-evaluate-model: evaluate trained models against the test set.</li> 
</ul>


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
