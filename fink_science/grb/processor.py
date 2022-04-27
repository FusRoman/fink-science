import string
from matplotlib import units
import pandas as pd
import numpy as np

from utilities import get_last_ztf_alert
from grb_catalog import get_notice, get_grb_in_tw
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
from astropy.coordinates import search_around_sky
import astropy.units as u

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import *  # noqa: F403
from pyspark.sql import functions as F  # noqa: F401
from pyspark.sql import SparkSession  # noqa: F401

import requests
import time as t

import datetime
import json

from grb_assoc import grb_associations


def grb_associations(ra: list, dec: list, jdstarthist: list, objectId: list, grb_bc: pd.DataFrame, monitor: string, grb_tw: int)-> pd.DataFrame:
    """
    Associates ztf objects and grb events by temporal and spatial cross-match.


    """
    day_sec = 24 * 3600 * grb_tw
    bottomlimit_window = Time.now().jd - TimeDelta(day_sec, format="sec").jd

    grb_schema = StructType(
        [
            StructField("tags", StringType(), True),
            StructField("grb_association", ArrayType(ArrayType(DoubleType())), True),
        ]
    )

    @pandas_udf(grb_schema)
    def aux_grb_score(ra, dec, jdstarthist, objectId):

        grb_notices = grb_bc.value

        ztf_pdf = pd.DataFrame(
            {"ra": ra, "dec": dec, "jdstarthist": jdstarthist, "objectId": objectId}
        )

        grb_res_assoc = ztf_pdf.apply(
            lambda x: grb_associations(x, grb_notices, monitor, bottomlimit_window),
            axis=1,
            result_type="expand",
        ).rename({0: "tags", 1: "grb_association"}, axis=1)

        return grb_res_assoc

    return aux_grb_score(ra, dec, jdstarthist, objectId)


if __name__ == "__main__":

    print("get grb notices")

    monitor = "fermi"
    grb_tw = 5

    grb_notices = get_notice(monitor)

    grb_notices = get_grb_in_tw(grb_notices, monitor, grb_tw)
    print(grb_notices)

    # load alerts from HBase, best way to do

    path_to_catalog = "hbase_catalogs/ztf_season1.class.json"
    path_to_catalog = (
        "/home/julien.peloton/fink-broker/ipynb/hbase_catalogs/ztf_season1.class.json"
    )

    with open(path_to_catalog) as f:
        catalog = json.load(f)

    master_adress = "mesos://vm-75063.lal.in2p3.fr:5050"

    spark = (
        SparkSession.builder.master(master_adress).appName("grb_module").getOrCreate()
    )

    df = (
        spark.read.option("catalog", catalog)
        .format("org.apache.hadoop.hbase.spark")
        .option("hbase.spark.use.hbasecontext", False)
        .option("hbase.spark.pushdown.columnfilter", True)
        .load()
    )

    request_class = ["SN candidate", "Ambiguous", "Unknown", "Solar System candidate"]

    now = Time.now().jd
    yesterday = now - TimeDelta(3600 * 36, format="sec").jd
    class_pdf = []

    for _class in request_class:
        class_pdf.append(
            df.filter(
                df["class_jd_objectId"] >= "{}_{}".format(_class, yesterday)
            ).filter(df["class_jd_objectId"] < "{}_{}".format(_class, now))
        )

    grb_bc = spark.sparkContext.broadcast(grb_notices)

    p_grb_list = []
    for alert_class in class_pdf:

        alert_class = alert_class.repartition(1000)

        p_grb = alert_class.withColumn(
            "grb_score",
            grb_associations(
                alert_class.ra,
                alert_class.dec,
                alert_class.jdstarthist,
                alert_class.objectId,
                grb_bc,
                monitor,
                grb_tw
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

    
    print([sc_p_grb.rdd.getNumPartitions() for sc_p_grb in p_grb_list])
    res_grb = pd.concat([sc_p_grb.toPandas() for sc_p_grb in p_grb_list])

    print(res_grb)