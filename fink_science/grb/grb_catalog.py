import string
import pandas as pd
import datetime


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
    elif grb_monitor == "swift":
        trigger_url = "https://gcn.gsfc.nasa.gov/swift_grbs.html"

    else:
        raise ValueError(
            "bad grb_monitor, must be either 'fermi' or 'swift', not {}".format(
                grb_monitor
            )
        )

    return pd.read_html(trigger_url)[0]


def get_grb_in_tw(
    grb_data: pd.DataFrame, monitor: string, start_window: datetime, time_window: int
) -> pd.DataFrame:
    """
    Restrict the grb catalog to keep only the grb in the time window. The time window is set to be
    between the time window parameters and the start_window.
    All the grb that respect {(start_window - time_window) < trigger time < start_window} are return.

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

    Returns
    -------
    grb_data : DataFrame
        the grb catalog with only the grb in the time window
    """

    if monitor == "fermi":
        # remove bogus date (like 22/01/-4918)
        grb_data = grb_data[
            ~grb_data[("TRIGGER", "Date")].str.match("(\d\d/){2}-(\d)*")
        ]
        with pd.option_context("mode.chained_assignment", None):
            grb_data["Trig Time"] = (
                grb_data[("TRIGGER", "Date")] + " " + grb_data[("TRIGGER", "Time UT")]
            )

    if monitor == "swift":
        test_date = (
            grb_data[("GRB/TRIGGER", "Date yy/mm/dd")]
            .str.fullmatch("(\d\d/){2}\d\d")
            .astype(bool)
        )
        grb_data = grb_data[test_date]

        with pd.option_context("mode.chained_assignment", None):
            grb_data["Trig Time"] = (
                grb_data[("GRB/TRIGGER", "Date yy/mm/dd")]
                + " "
                + grb_data[("GRB/TRIGGER", "Time UT")]
            )

    with pd.option_context("mode.chained_assignment", None):
        grb_data["Trig Time"] = pd.to_datetime(grb_data["Trig Time"], yearfirst=True)

    # compute the bottom limit
    bottomlimit_window = start_window - datetime.timedelta(days=time_window)

    grb_data = grb_data[
        (grb_data["Trig Time"] >= bottomlimit_window)
        & (grb_data["Trig Time"] <= start_window)
    ].reset_index(drop=True)
    return grb_data
