import string
import numpy as np
import pandas as pd
import requests

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

from grb_calc import p_ser_grb_vect, sig_est, compute_rate
from utilities import request_fink


def grb_notices_getter(monitor: string) -> tuple:
    """
    Return the column names of the grb catalog corresponding to the given monitor.
    Return also the grb trigger rate of the corresponding monitor (in trig/year).

    Parameters
    ----------
    monitor : string
        grb monitor, must be 'fermi' or 'swift'

    Returns
    -------
    columns_names : tuple
        The corresponding columns names for the GRB Trigger number, ra, dec, the grb error box
         and the monitor trigger rate.
    """

    if monitor == "fermi":

        return (
            ("TRIGGER", "TrigNum"),
            ("OBSERVATION", "RA(J2000)[deg]"),
            ("OBSERVATION", "Dec(J2000)[deg]"),
            ("OBSERVATION", "Error[deg][arcmin]"),
            250,
        )

    elif monitor == "swift":

        return (
            ("GRB/TRIGGER", "Trig"),
            ("OBSERVATION", "BAT RA"),
            ("OBSERVATION", "BAT Dec"),
            ("OBSERVATION", "BAT Error"),
            100,
        )

    else:
        raise ValueError("monitor must be 'fermi' of 'swift', not {}".format(monitor))


def cross_match_space_and_time(
    ztf_rows: pd.Series,
    grb_notices: pd.DataFrame,
    monitor: string,
    start_tw: float,
    bottomlimit_window: float,
) -> tuple:
    """
    Associates a ztf object with one or more grb events.

    Parameters
    ----------
    ztf_rows : pd.Series
        the ztf objects with the corresponding columns:
            - ra
            - dec
            - jdstarthist
            - objectId
    grb_notices : dataframe
        the grb catalog
    monitor : string
        the grb monitor corresponding to the catalog
    start_tw: the start date of the searching time window, in julian date.
    bottomlimit_window : float
        the bottom limit of the time window to filter late alerts, in julian date.

    Returns
    -------
    grb_associations : tuple
        A tuple containing the informations of the associations :
            - firstly, a tag that indicates what process has produced the associations
                'proba-based-cross-match' -> the grb has been associated by probability.
                'fast-transient-based-cross-match' -> the grb has been associated because
                    the ztf alerts fall within the grb error box, has a low associations
                    probability but behave like a fast transient.
            - secondly, for the ztf object associated with a grb events by probability,
                one return all the following informations :
                    - trigger number of the grb event
                    - probability that the ztf object and grb event can be associated in general
                    - sigma error of the above probability
                    - probability that the ztf object can ben associated with a long grb.
                    - sigma error of the above probability
                    - probability that the ztf object can ben associated with a short grb.
                    - sigma error of the above probability
                As a ztf alert can fall within multiples error boxes if two or more grb events occurs closely,
                one return a list for all of the above items

        In case of non associations, one return the 'No match' tag and empty lists for the probabilities.
    """

    (
        select_grb_num,
        select_grb_ra,
        select_grb_dec,
        select_grb_error,
        grb_det_rate,
    ) = grb_notices_getter(monitor)

    # remove ztf alerts with a start variation time far away from the beginning of the grb time window
    if (ztf_rows["jdstarthist"] >= bottomlimit_window) and (ztf_rows["jd"] <= start_tw):

        grb_coord = SkyCoord(
            grb_notices[select_grb_ra], grb_notices[select_grb_dec], unit=u.degree
        )

        ztf_coords = SkyCoord(ztf_rows["ra"], ztf_rows["dec"], unit=u.degree)

        grb_error_box = grb_notices[select_grb_error]

        # test if some ztf alerts fall in the error box of the grb

        ## 1.5 * grb_error_box
        ## en fonction du monitor, changer les unités : fermi en degrées et swift en arcminute
        test_assoc_grb = (
            ztf_coords.separation(grb_coord).degree < grb_error_box * u.degree * 1.5
        )
        sep_associated_grb = grb_notices[test_assoc_grb]

        # keep only the grb with a ztf alerts that fall within the error box
        if len(sep_associated_grb) > 0:

            # compute the delay between the grb and the ztf alerts
            grb_trig = Time(sep_associated_grb["Trig Time"], format="datetime64").jd
            delay = ztf_rows["jdstarthist"] - grb_trig
            sep_associated_grb["delay"] = delay
            time_associated_grb = sep_associated_grb[delay > 0]

            # keep only the ztf alerts with a start variation time after the grb emission
            if len(time_associated_grb) > 0:
                grb_id = time_associated_grb[select_grb_num].values
                grb_error_box = time_associated_grb[select_grb_error].values

                # convert the delay from day in year
                delay_year = time_associated_grb["delay"] / 365.25

                # compute the serendipitous probability for each ztf alerts to be associated with the current grb
                p_ser = p_ser_grb_vect(grb_error_box, delay_year, grb_det_rate)

                # compute the sigma error estimation of the probability for grb in general
                sigma_p_ser = [sig_est(1 - p)[0] for p in p_ser[0] if len(p_ser[0]) > 0]

                sigma_p_ser_array = np.array(sigma_p_ser)

                # filter the alerts with a sigma error on the grb probability above 5
                cut_grb = sigma_p_ser_array > 5
                if np.any(cut_grb):

                    grb_id = grb_id[cut_grb]

                    # compute the sigma error estimation of the probability for long grb
                    sigma_lp_ser = [
                        sig_est(1 - p)[0]
                        for p in p_ser[1][cut_grb]
                        if len(p_ser[1]) > 0
                    ]

                    # compute the sigma error estimation of the probability for short grb
                    sigma_sp_ser = [
                        sig_est(1 - p)[0]
                        for p in p_ser[2][cut_grb]
                        if len(p_ser[2]) > 0
                    ]

                    return "proba-based-cross-match", [
                        grb_id.tolist(),
                        p_ser[0][cut_grb].tolist(),
                        sigma_p_ser_array[cut_grb].tolist(),
                        p_ser[1][cut_grb].tolist(),
                        sigma_lp_ser,
                        p_ser[2][cut_grb].tolist(),
                        sigma_sp_ser,
                    ]
                else:
                    objectid = ztf_rows["objectId"]
                    alert_history = request_fink(objectid)

                    # search for the detection of fast transient in the error box of the grb.
                    mag_test = compute_rate(alert_history)
                    if mag_test:
                        return "fast-transient-based-cross-match", [
                            grb_id.tolist(),
                            # ajouter la proba p_ser
                            p_ser[0][cut_grb].tolist(),
                            sigma_p_ser_array[cut_grb].tolist(),
                            [],
                            [],
                            [],
                            [],
                        ]
                    else:
                        return "Not behave like a fast transient", []

            else:
                return "Before the GRB trigger time", []

        else:
            return "Not in any GRB error box", []

    else:
        return "Start variation time not in the GRB time window", []
