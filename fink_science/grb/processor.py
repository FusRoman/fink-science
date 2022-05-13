import re
import string
import pandas as pd
import json
from datetime import datetime
from datetime import timedelta

from astropy.time import Time, TimeDelta

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *  # noqa: F403
from pyspark.sql import functions as F  # noqa: F401
from pyspark.sql import SparkSession  # noqa: F401

from fink_utils.xmatch.simbad import return_list_of_eg_host

from grb_assoc import cross_match_space_and_time, grb_notices_getter
from grb_catalog import get_notice, get_grb_in_tw


def grb_associations(
    ra: list,
    dec: list,
    jd: list,
    jdstarthist: list,
    objectId: list,
    grb_bc: pd.DataFrame,
    monitor: string,
    start_grb_tw: datetime,
    grb_tw: int,
) -> pd.DataFrame:
    """
    Associates ztf objects and grb events by temporal and spatial cross-match.

    Parameters
    ----------
    ra : list
        right ascension of alerts
    dec : list
        declination of alerts
    jd : list
        start exposure time of the alerts
    jdstarthist : list
        first time that the alerts begun to variates
    objectId : list
        object identifier of the alerts
    grb_bc : dataframe
        the grb events catalog send to executor with a broadcast
    monitor : string
        the name of the grb monitor used to download the catalog
    start_grb_tw : datetime
        The start date of the time window, all the ztf alerts that respect jdstarthist >= (start_grb_tw - grb_tw) and jd <= start_grb_tw
        are tested for the association with a grb.
    grb_tw : integer
        the bottom limit of the time window, in days

    Return
    ------
    grb_associations : dataframe
        A dataframe containing the informations of the associations :
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
    day_sec = 24 * 3600 * grb_tw
    start_wd_jd = Time(start_grb_tw, scale="utc").jd
    bottomlimit_window = start_wd_jd - TimeDelta(day_sec, format="sec").jd

    grb_schema = StructType(
        [
            StructField("tags", StringType(), True),
            StructField("grb_association", ArrayType(ArrayType(DoubleType())), True),
        ]
    )

    @pandas_udf(grb_schema)
    def aux_grb_score(ra, dec, jd, jdstarthist, objectId):
        """
        see the documentation of the main function
        """
        grb_notices = grb_bc.value

        ztf_pdf = pd.DataFrame(
            {
                "ra": ra,
                "dec": dec,
                "jd": jd,
                "jdstarthist": jdstarthist,
                "objectId": objectId,
            }
        )

        grb_res_assoc = ztf_pdf.apply(
            lambda x: cross_match_space_and_time(
                x, grb_notices, monitor, start_wd_jd, bottomlimit_window
            ),
            axis=1,
            result_type="expand",
        ).rename({0: "tags", 1: "grb_association"}, axis=1)

        return grb_res_assoc

    return aux_grb_score(ra, dec, jd, jdstarthist, objectId)


def detect_grb_counterparts(grb_notices, monitor, start_window, grb_window_width):

    grb_notices = get_grb_in_tw(grb_notices, monitor, start_window, grb_window_width)

    print(grb_notices[[("TRIGGER", "TrigNum"), ("OBSERVATION", "RA(J2000)[deg]"), ("OBSERVATION", "Dec(J2000)[deg]")]])

    print()
    print()

    # cast the columns to numeric types
    get_num_cols = list(grb_notices_getter(monitor))[:-1]
    grb_notices[get_num_cols] = grb_notices[get_num_cols].apply(pd.to_numeric)

    print("nb grb in the time window : {}".format(len(grb_notices)))

    if len(grb_notices) == 0:
        return pd.DataFrame(
            columns=[
                "jd",
                "jdstarthist",
                "ra",
                "dec",
                "objectId",
                "tags",
                "GRB_trignum",
                "p_ser",
                "sigma_p_ser",
                "lp_ser",
                "sigma_lp_ser",
                "sp_ser",
                "sigma_sp_ser",
            ]
        )

    # load alerts from HBase, best way to do

    path_to_catalog = "hbase_catalogs/ztf_season1.class.json"
    path_to_catalog = (
        "/home/julien.peloton/fink-broker/ipynb/hbase_catalogs/ztf_season1.class.json"
    )

    with open(path_to_catalog) as f:
        catalog = json.load(f)

    master_adress = "mesos://vm-75063.lal.in2p3.fr:5050"

    spark = (
        SparkSession.builder.master(master_adress).appName("grb_module_{}".format(monitor)).getOrCreate()
    )

    spark.sparkContext.setLogLevel("FATAL")

    t_before = t.time()
    df = (
        spark.read.option("catalog", catalog)
        .format("org.apache.hadoop.hbase.spark")
        .option("hbase.spark.use.hbasecontext", False)
        .option("hbase.spark.pushdown.columnfilter", True)
        .load()
        .repartition(500)
    )
    print("loading time: {}".format(t.time() - t_before))
    print()

    request_class = [# return_list_of_eg_host() + [
        # "Ambiguous",
        # "Solar System candidate",
        "SN candidate",
    ]

    start_wd_astropy = Time(start_window, scale="utc").jd
    yesterday = start_wd_astropy - TimeDelta(3600 * 36, format="sec").jd
    class_pdf = []

    for _class in request_class:
        class_pdf.append(
            df.filter(
                df["class_jd_objectId"] >= "{}_{}".format(_class, yesterday)
            ).filter(df["class_jd_objectId"] < "{}_{}".format(_class, start_wd_astropy))
        )

    # send the grb data to the executors
    grb_bc = spark.sparkContext.broadcast(grb_notices)

    p_grb_list = []
    for alert_class in class_pdf:

        if alert_class.count() == 0:
            continue

        alert_class = alert_class

        p_grb = alert_class.withColumn(
            "grb_score",
            grb_associations(
                alert_class.ra,
                alert_class.dec,
                alert_class.jd,
                alert_class.jdstarthist,
                alert_class.objectId,
                grb_bc,
                monitor,
                start_window,
                grb_window_width,
            ),
        )

        p_grb = p_grb.select(
            p_grb.jd,
            p_grb.jdstarthist,
            p_grb.ra,
            p_grb.dec,
            p_grb.objectId,
            p_grb.grb_score["tags"].alias("tags"),
            p_grb.grb_score["grb_association"][0]
            .alias("GRB_trignum")
            .cast(ArrayType(IntegerType())),
            p_grb.grb_score["grb_association"][1].alias("p_ser"),
            p_grb.grb_score["grb_association"][2].alias("sigma_p_ser"),
            p_grb.grb_score["grb_association"][3].alias("lp_ser"),
            p_grb.grb_score["grb_association"][4].alias("sigma_lp_ser"),
            p_grb.grb_score["grb_association"][5].alias("sp_ser"),
            p_grb.grb_score["grb_association"][6].alias("sigma_sp_ser"),
        )
        p_grb_list.append(p_grb)

    if len(p_grb_list) == 0:
        return pd.DataFrame(
            columns=[
                "jd",
                "jdstarthist",
                "ra",
                "dec",
                "objectId",
                "tags",
                "GRB_trignum",
                "p_ser",
                "sigma_p_ser",
                "lp_ser",
                "sigma_lp_ser",
                "sp_ser",
                "sigma_sp_ser",
            ]
        )

    t_before = t.time()
    res_grb = pd.concat([sc_p_grb.toPandas() for sc_p_grb in p_grb_list])
    print("spark jobs total time: {}".format(t.time() - t_before))

    return res_grb


if __name__ == "__main__":

    import time as t
    from collections import Counter
    import sys

    monitor = sys.argv[1]
    print("start grb module with the {} monitor".format(monitor))
    print()
    
    start_window = datetime.fromisoformat("2021-02-05")

    dt = timedelta(days=1)
    grb_tw = 5

    grb_notices = get_notice(monitor)

    for _ in range(10):

        print("current date: {}".format(start_window))
        print()
        grb_counterparts = detect_grb_counterparts(
            grb_notices, monitor, start_window, grb_tw
        )

        print(grb_counterparts)
        print()
        print()

        print(Counter(grb_counterparts["tags"]))
        print()
        print()

        ft_objId = grb_counterparts[grb_counterparts["tags"] == "fast-transient-based-cross-match"]

        proba_objId = grb_counterparts[grb_counterparts["tags"] == "proba-based-cross-match"]["objectId"]

        print(grb_counterparts[grb_counterparts["objectId"].isin(proba_objId)][["objectId", "GRB_trignum", "p_ser", "sigma_p_ser"]])

        print()
        print("BINGO !!!!")
        print(ft_objId[ft_objId["objectId"] == "ZTF21aagwbjr"])
        print()
        print()

        start_window += dt
