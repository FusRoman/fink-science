import numpy as np
import pandas as pd

from math import pi, sqrt
from scipy.stats import poisson
from scipy import special
from scipy.optimize import minimize

from fink_utils.photometry.conversion import dc_mag


def p_ser_grb_vect(
    error_radius: np.array, size_time_window: np.array, r_grb: int
) -> list:
    """
    Created on Mon Oct  4 10:34:09 2021

    @author: Damien Turpin : damien.turpin@cea.fr

    function that gives the chance probability of having a positive spatial and
    temporal match between a GRB and a ZTF transient candidate

    Parameters
    ----------
    error_radius : array
        error radius of the GRB localization region in degree
    size_time_window: array
        size of the searching time window in year
    r_grb: integer
        GRB detection rate for a given satellite in events/year

    Returns
    -------
    p_ser : list
        Serendipituous probabilities for a GRB/ZTF candidate association.
        The first items correspond to the association probability with a GRB in general, the second correspond
        to the association with a long GRB and finally, the last items correspond to the associations with a
        short GRB.
    """

    # omega = 2*pi*(1-cos(radians(error_radius))) # solid angle in steradians
    grb_loc_area = pi * (error_radius) ** 2  # in square degrees
    allsky_area = 4 * pi * (180 / pi) ** 2  # in square degrees
    ztf_coverage_rate = 3750  # sky coverage rate of ZTF in square degrees per hour
    limit_survey_time = 4  # duration (in hour) during which ZTF will cover individual parts of the sky in a night

    # short and long GRB detection rate
    r_sgrb = r_grb / 3
    r_lgrb = r_grb - r_sgrb

    # Poisson probability of detecting a GRB during a searching time window

    p_grb_detect_ser = 1 - poisson.cdf(1, r_grb * size_time_window)
    p_lgrb_detect_ser = 1 - poisson.cdf(1, r_lgrb * size_time_window)
    p_sgrb_detect_ser = 1 - poisson.cdf(1, r_sgrb * size_time_window)

    # we limit the fraction of the sky ZTF is able to cover to 4 hours of continuous survey
    # we consider that every day (during several days only) ZTF will cover the same part of
    # the sky with individual shots (so no revisit) during 4 hours

    #     if size_time_window*365.25*24 <= limit_survey_time:
    #         ztf_sky_frac_area = (ztf_coverage_rate*size_time_window*365.25*24)
    #     else:
    #         ztf_sky_frac_area = ztf_coverage_rate*limit_survey_time

    ztf_sky_frac_area = np.where(
        size_time_window * 365.25 * 24 <= limit_survey_time,
        (ztf_coverage_rate * size_time_window * 365.25 * 24),
        ztf_coverage_rate * limit_survey_time,
    )

    # probability of finding a GRB within the region area paved by ZTF during a given amount of time
    p_grb_in_ztf_survey = (ztf_sky_frac_area / allsky_area) * p_grb_detect_ser
    p_lgrb_in_ztf_survey = (ztf_sky_frac_area / allsky_area) * p_lgrb_detect_ser
    p_sgrb_in_ztf_survey = (ztf_sky_frac_area / allsky_area) * p_sgrb_detect_ser

    # probability of finding a ZTF transient candidate inside the GRB error box
    # knowing the GRB is in the region area paved by ZTF during a given amount of time

    p_ser_grb = p_grb_in_ztf_survey * (grb_loc_area / ztf_sky_frac_area)

    p_ser_lgrb = p_lgrb_in_ztf_survey * (grb_loc_area / ztf_sky_frac_area)

    p_ser_sgrb = p_sgrb_in_ztf_survey * (grb_loc_area / ztf_sky_frac_area)

    p_sers = [p_ser_grb, p_ser_lgrb, p_ser_sgrb]

    return p_sers


def sig_est(prob: float) -> float:
    """
    Created on Mon Oct  4 10:34:09 2021

    @author: Damien Turpin : damien.turpin@cea.fr

    Give an estimation of the standard deviation of the GRB association probability.

    Parameters
    ----------
    prob : float
        The GRB association probability

    Returns
    -------
    res.x : float
        standard deviation estimation
    """
    fun = lambda x: abs(prob - special.erf(x / sqrt(2)))
    res = minimize(fun, [0], method="Nelder-Mead")

    return res.x


def compute_rate(ztf_rows: pd.Series) -> bool:
    """
    Compute the magnitude rates of a ztf objects in order to detect a fast transient.

    The mag rate are computed between the valid detection
    and between the first valid detection and the last upper limit.


    """

    mag_rate = []

    cfid = np.array(ztf_rows["cfid"].values[0])
    cjd = np.array(ztf_rows["cjd"].values[0])
    cdiffmaglim = np.array(ztf_rows["cdiffmaglim"].values[0])

    # compute apparent magnitude of the alerts history
    mag_res = np.array(
        [
            dc_mag(i[0], i[1], i[2], i[3], i[4], i[5], i[6])[0] if i[1] != None else -1
            for i in zip(
                cfid,
                ztf_rows["cmagpsf"].values[0],
                ztf_rows["csigmapsf"].values[0],
                ztf_rows["cmagnr"].values[0],
                ztf_rows["csigmagnr"].values[0],
                ztf_rows["cmagzpsci"].values[0],
                ztf_rows["cisdiffpos"].values[0],
            )
        ]
    )

    # for each ztf filters
    for ztf_filt in np.unique(cfid):

        # get the index of the corresponding filter
        filter_idx = np.where(cfid == ztf_filt)

        diffmaglim_filter = cdiffmaglim[filter_idx]
        mag_filter = mag_res[filter_idx]
        jd_filter = cjd[filter_idx]

        # compute the mag rate between the first valid magnitude value and the last upper-limit
        first_mag_idx = np.where(mag_filter != -1)[0]
        if len(first_mag_idx) > 0:
            diff_time = jd_filter[first_mag_idx[0] - 1] - jd_filter[first_mag_idx[0]]
            diff_mag = (
                diffmaglim_filter[first_mag_idx[0] - 1] - mag_filter[first_mag_idx[0]]
            )

            if diff_time < 1e-3:
                with_ul_rate = diff_mag
            else:
                with_ul_rate = diff_mag / diff_time

            mag_rate.append(with_ul_rate)

        # compute the mag rate of the valid values
        valid_mag_rate = np.diff(mag_filter[mag_filter != -1]) / np.diff(
            jd_filter[mag_filter != -1]
        )
        mag_rate += valid_mag_rate.tolist()

    return np.any(np.abs(mag_rate) > 0.3)
