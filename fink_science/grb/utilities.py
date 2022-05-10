import string
import pandas as pd
import requests


def request_fink(objectid: string) -> pd.DataFrame:
    """
    Retrieve the history of an object.

    Parameters
    ----------
    objectid: string
        The object identifier of the requested object

    Returns
    -------
    object_history : dataframe
        The history of the object
    """
    r = requests.post(
        "https://fink-portal.org/api/v1/objects",
        json={
            "objectId": objectid,
            "output-format": "json",
            "withupperlim": True,
            "withcutouts": False,
            "columns": "i:objectId,i:fid,i:jd,i:magpsf,i:sigmapsf,i:magnr,i:sigmagnr,i:magzpsci,i:isdiffpos",
        },
    )

    # Format output in a DataFrame
    pdf = pd.read_json(r.content)

    object_history = pdf.groupby(["i:objectId"]).agg(
        cjd=("i:jd", list),
        cfid=("i:fid", list),
        cmagpsf=("i:magpsf", list),
        csigmapsf=("i:sigmapsf", list),
        cmagnr=("i:magnr", list),
        csigmagnr=("i:sigmagnr", list),
        cmagzpsci=("i:magzpsci", list),
        cisdiffpos=("i:isdiffpos", list),
        cdiffmaglim=("i:diffmaglim", list),
    )

    return object_history
