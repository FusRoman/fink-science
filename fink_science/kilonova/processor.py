# Copyright 2021 AstroLab Software
# Author: Julien Peloton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType, StringType

import pandas as pd
import numpy as np

import os

from fink_science.conversion import mag2fluxcal_snana
from fink_science.utilities import load_scikit_model, load_pcs

from fink_science import __file__

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def knscore(jd, fid, magpsf, sigmapsf, model_path=None, pcs_path=None, npcs=None) -> pd.Series:
    """ Return the probability of an alert to be a Kilonova using a Random
    Forest Classifier.

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    model_path: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/model_3PC_2KN_Cosmins_models.pkl` is loaded.
    pcs_path: Spark DataFrame Column, optional
        Path to the Principal Component file. Default is None, in which case
        the `data/models/components.csv` is loaded.
    npcs: Spark DataFrame Column, optional
        Integer representing the number of Principal Component to use. It
        should be consistent to the training model used. Default is None (i.e.
        default npcs for the default `model_path`, that is 3).

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-KNe) and 1 (KNe).

    Examples
    ----------
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> df = df.withColumn('pKNe', knscore(*args))

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in what_prefix] + \
    >>>   [F.lit(model_path), F.lit(comp_path), F.lit(1)]
    >>> df = df.withColumn('pKNe', knscore(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.agg({"pKNe": "min"}).collect()[0][0]
    0.0

    >>> df.agg({"pKNe": "max"}).collect()[0][0] < 1.0
    True
    """
    epoch_lim = [-50, 50]
    time_bin = 0.25
    flux_lim = 0
    norm = False
    npcs = int(npcs.values[0])

    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 1
    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    # add an exploded column with SNID
    df_tmp = pd.DataFrame.from_dict(
        {
            'jd': jd[mask],
            'SNID': range(len(jd[mask]))
        }
    )
    df_tmp = df_tmp.explode('jd')

    # compute flux and flux error
    data = [mag2fluxcal_snana(*args) for args in zip(
        magpsf[mask].explode(),
        sigmapsf[mask].explode())]
    flux, error = np.transpose(data)

    # make a Pandas DataFrame with exploded series
    pdf = pd.DataFrame.from_dict({
        'SNID': df_tmp['SNID'],
        'MJD': df_tmp['jd'],
        'FLUXCAL': flux,
        'FLUXCALERR': error,
        'FLT': fid[mask].explode().replace({1: 'g', 2: 'r'})
    })

    # Load pre-trained model `clf`
    if model_path is not None:
        model = load_scikit_model(model_path.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model_path = curdir + '/data/models/model_1PC_2KN_Cosmins_models.pkl'
        model = load_scikit_model(model_path)

    # Load pcs
    if pcs_path is not None:
        pcs = load_pcs(pcs_path.values[0], npcs)
    else:
        # default - 1 component
        curdir = os.path.dirname(os.path.abspath(__file__))
        pcs_path_ = curdir + '/data/models/components.csv'
        pcs = load_pcs(pcs_path_, npcs=1)

    test_features = []
    filters = ['g', 'r']

    # extract features (all filters) for each ID
    for id in np.unique(pdf['SNID']):
        pdf_sub = pdf[pdf['SNID'] == id]
        pdf_sub = pdf_sub[pdf_sub['FLUXCAL'] == pdf_sub['FLUXCAL']]
        features = extract_all_filters_fink(
            epoch_lim=epoch_lim, pcs=pcs,
            time_bin=time_bin, filters=filters,
            lc=pdf_sub, flux_lim=flux_lim, norm=False)
        test_features.append(features)

    # Remove pathological values
    names_root = [
        'npoints_',
        'residuo_'
    ] + [
        'coeff' + str(i + 1) + '_' for i in range(len(pcs.keys()))
    ]

    columns = [i + j for j in ['g', 'r'] for i in names_root]

    matrix = pd.DataFrame(test_features, columns=columns)

    zeros = np.logical_or(
        matrix['coeff1_g'].values == 0,
        matrix['coeff1_r'].values == 0
    )

    matrix_clean = matrix[~zeros]

    # Make predictions
    probabilities = model.predict_proba(matrix_clean.values)
    probabilities_notkne = np.zeros(len(test_features))
    probabilities_kne = np.zeros(len(test_features))

    probabilities_notkne[~zeros] = probabilities.T[0]
    probabilities_kne[~zeros] = probabilities.T[1]
    probabilities_ = np.array([probabilities_notkne, probabilities_kne]).T

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities_.T[1]

    return pd.Series(to_return)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/alerts.parquet'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path = '{}/data/models/model_1PC_2KN_Cosmins_models.pkl'.format(path)
    globs["model_path"] = model_path

    comp_path = '{}/data/models/components.csv'.format(path)
    globs["comp_path"] = comp_path

    # Run the test suite
    spark_unit_tests(globs)