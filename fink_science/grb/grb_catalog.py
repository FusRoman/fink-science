from os import rename
import string
import pandas as pd
import numpy as np
import datetime
import astropy.units as u


def get_notice(grb_monitor: string) -> pd.DataFrame:
    """
    Retrieve the grb catalog corresponding to the grb_monitor given in parameters

    Parameters
    ----------
    grb_monitor : string
        grb_monitor correponding to the catalog that we want. Must be either 'fermi' or 'swift'

    Returns
    -------
    catalog : DataFrame
        the grb catalog taken from the corresponding website
    """

    if grb_monitor == "fermi":
        trigger_url = "https://gcn.gsfc.nasa.gov/fermi_grbs.html"

        renaming_dict = {"TrigNum" : "TrigNum",
            "Date" : "date",
            "Time UT" : "time_ut",
            "RA(J2000)[deg]" : "gm_ra",
            "Dec(J2000)[deg]" : "gm_dec",
            "Error[deg][arcmin]" : "gm_error"
        }

    elif grb_monitor == "swift":
        trigger_url = "https://gcn.gsfc.nasa.gov/swift_grbs.html"

        renaming_dict = {"Trig" : "TrigNum",
            "Date yy/mm/dd" : "date",
            "Time UT" : "time_ut",
            "BAT RA" : "gm_ra",
            "BAT Dec" : "gm_dec",
            "BAT Error" : "gm_error",
            "XRT RA" : "xrt_ra",
            "XRT Dec" : "xrt_dec",
            "XRT Error" : "xrt_error"
        }

    else:
        raise ValueError(
            "bad grb_monitor, must be either 'fermi' or 'swift', not {}".format(
                grb_monitor
            )
        )

    grb_notices = pd.read_html(trigger_url)[0]

    grb_notices.columns = grb_notices.columns.droplevel()

    keep_cols = list(renaming_dict.values())

    return grb_notices.rename(renaming_dict, axis='columns')[keep_cols]


def grb_filter(
    grb_data: pd.DataFrame, monitor: string, start_window: datetime, time_window: int, error_limit: float=10
) -> pd.DataFrame:
    """
    Restrict the grb catalog to keep only the grb in the time window. The time window is set to be
    between the time window parameters and the start_window.
    All the grb that respect {(start_window - time_window) <= trigger time < start_window} are return.

    Parameters
    ----------
    grb_data : DataFrame
        the grb catalog
    monitor : string
        the monitor corresponding to the catalog
    start_window : datetime
        the start date of the time window
    time_window : int
        the time window in day corresponding to the bottom limit. The bottom limit of the time window are computed as (start_window - time_window)
    error_limit : float
        remove the grb with an error box greater than the limit, only significant for the fermi monitor

    Returns
    -------
    grb_data : DataFrame
        the grb catalog with only the grb in the time window
    """

    if monitor == "fermi":
        # remove bogus date (like 22/01/-4918)
        grb_data = grb_data[
            ~grb_data["date"].str.match("(\d\d/){2}-(\d)*")
        ]

    if monitor == "swift":
        test_date = (
            grb_data["date"]
            .str.fullmatch("(\d\d/){2}\d\d")
            .astype(bool)
        )
        grb_data = grb_data[test_date]


    with pd.option_context("mode.chained_assignment", None):
        grb_data["Trig Time"] = (
            grb_data["date"]
            + " "
            + grb_data["time_ut"]
        )

    with pd.option_context("mode.chained_assignment", None):
        grb_data["Trig Time"] = pd.to_datetime(grb_data["Trig Time"], yearfirst=True)

    # compute the bottom limit
    bottomlimit_window = start_window - datetime.timedelta(days=time_window)

    grb_data = grb_data[
        (grb_data["Trig Time"] >= bottomlimit_window)
        & (grb_data["Trig Time"] <= start_window)
    ].reset_index(drop=True)

    # keep the last grb notices (especially for Fermi, the error box of the latest notices are usually the best). 
    grb_data = grb_data.drop_duplicates(["TrigNum"], keep="first")

    if monitor == "swift":
        grb_data["precision"] = np.where(~pd.isna(grb_data["xrt_error"]), u.arcsecond, u.arcminute)
        x_cols = ["xrt_ra", "xrt_dec", "xrt_error"]
        gm_cols = ["gm_ra", "gm_dec", "gm_error"]
        for x_col, gm_col in zip(x_cols, gm_cols):
            grb_data[x_col] = grb_data[x_col].fillna(grb_data[gm_col])

    if monitor == "fermi":
        # filter the grb with a large error box
        grb_data = grb_data[grb_data["gm_error"] < error_limit]


    return grb_data


if __name__ == "__main__":

    fermi_gcn = get_notice("fermi")

    # swift_gcn = get_notice("swift")

    start_window = datetime.datetime.fromisoformat("2021-02-06")
    grb_tw = 5

    fermi_tw = grb_filter(fermi_gcn, "fermi", start_window, grb_tw, 10)

    # swift_tw = grb_filter(swift_gcn, "swift", start_window, grb_tw)

    print(fermi_tw)
    # print()
    # print(swift_tw)