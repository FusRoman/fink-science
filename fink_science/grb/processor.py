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


def grb_score(ra, dec, jdstarthist, objectId, grb_bc, monitor, grb_tw):

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
            grb_score(
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


    # print("get last grb notice")
    # t_before = t.time()
    # grb_notice = get_notice("fermi")
    # print("time taken to retrieve the grb notices: {}".format(t.time() - t_before))

    # print()

    # # remove bogus date (like 22/01/-4918)
    # grb_notice = grb_notice[~grb_notice[('TRIGGER', 'Date')].str.match('(\d\d/){2}-(\d)*')]

    # grb_notice["Trig Time"] = grb_notice[('TRIGGER', 'Date')] + " " + grb_notice[('TRIGGER', 'Time UT')]

    # grb_notice["Trig Time"] = pd.to_datetime(grb_notice["Trig Time"], yearfirst=True)
    # bottomlimit_window = datetime.datetime.now() - datetime.timedelta(days=10)
    # grb_notice = grb_notice[grb_notice["Trig Time"] > bottomlimit_window].reset_index(drop=True)

    # print("Nb grb in the time window: {}".format(len(grb_notice)))
    # print()

    # request_class = ["SN candidate","Ambiguous","Unknown", "Solar System candidate"]

    # t_before = t.time()
    # ztf_last_alert = get_last_ztf_alert(request_class, "2022-04-20", "i:objectId")
    # print("time taken to get the last objectId: {:.3f}".format(t.time() - t_before))
    # print()

    # last_seen_objects = ",".join(ztf_last_alert["i:objectId"].values)

    # t_before = t.time()
    # request_columns = "i:ra,i:dec,i:jd,i:nid,i:fid,i:magpsf,i:sigmapsf,i:magnr,i:sigmagnr,i:magzpsci,i:isdiffpos,i:objectId, i:jdstarthist"
    # r = requests.post(
    #     'https://fink-portal.org/api/v1/objects',
    #     json={
    #         'objectId': last_seen_objects,
    #         'output-format': 'json',
    #         'withupperlim': True,
    #         'withcutouts': False,
    #         'columns': request_columns
    #     }
    # )

    # # Important: sort by objectId and jd in order to correctly compute the delay with the grb
    # agregated_objects = pd.read_json(r.content).sort_values(["i:objectId", "i:jd"])

    # print("time taken to get the agregated light curves of the last objectId: {:.3f}".format(t.time() - t_before))
    # print()
    # print()

    # object_gb = agregated_objects.groupby(["i:objectId"]).agg({
    #     "i:jd": list,
    #     "i:sigmapsf": list,
    #     "i:ra": list,
    #     "i:dec": list,
    #     "i:jdstarthist":list
    # }).reset_index()

    # bottomlimit_tw_jd = Time(bottomlimit_window, format="datetime")

    # def compute_delay(agg_obj):
    #     idx_real_value = np.argwhere(np.logical_not(np.isnan(agg_obj["i:sigmapsf"])))
    #     return np.min(np.array(agg_obj["i:jd"])[idx_real_value]) - Time(bottomlimit_tw_jd, format="datetime").jd

    # object_gb["delay"] = object_gb.apply(compute_delay, axis=1)
    # object_gb = object_gb[object_gb["delay"] > 0]

    # object_gb["l_ra"] = object_gb["i:ra"].str[-1]
    # object_gb["l_dec"] = object_gb["i:dec"].str[-1]
    # object_gb["i:jdstarthist"] = object_gb["i:jdstarthist"].str[-1]

    # object_gb = object_gb.reset_index(drop=True)

    # def grb_associations(grb_rows, ztf_alerts):

    #     grb_coord = SkyCoord(
    #         grb_rows[("OBSERVATION", "RA(J2000)[deg]")],
    #         grb_rows[("OBSERVATION", "Dec(J2000)[deg]")],
    #         unit=u.degree
    #     )

    #     ztf_coords = SkyCoord(
    #         ztf_alerts["l_ra"],
    #         ztf_alerts["l_dec"],
    #         unit=u.degree
    #     )

    #     grb_error_box = grb_rows[("OBSERVATION", "Error[deg][arcmin]")]

    #     test_assoc_grb = ztf_coords.separation(grb_coord) < grb_error_box * u.degree

    #     for l in ztf_alerts.loc[test_assoc_grb, "assoc_grb"]:
    #         l.append(grb_rows[("TRIGGER", "TrigNum")])

    # object_gb['assoc_grb'] = [[] for _ in range(len(object_gb))]
    # grb_notice.apply(lambda x: grb_associations(x, object_gb), axis=1)
    # print(object_gb)
    # print()
    # print()

    # def compute_proba(ztf_rows):
    #     print(ztf_rows["i:objectId"])
    #     grb = grb_notice[grb_notice[("TRIGGER", "TrigNum")].isin(ztf_rows["assoc_grb"])]
    #     grb_trig = Time(grb["Trig Time"], format="datetime64").jd
    #     delay = ztf_rows["i:jdstarthist"] - grb_trig

    #     grb_error_box = grb[("OBSERVATION", "Error[deg][arcmin]")].values
    #     grb_id = grb[("TRIGGER", "TrigNum")].values
    #     res = []
    #     for d, e, g in zip(delay, grb_error_box, grb_id):
    #         if d > 0:
    #             p = p_ser_grb(e,d/365.25, 250)
    #             assoc_prob = sig_est(1-p[0])[0]
    #             res.append([g, assoc_prob])
    #     return res

    # object_gb["grb_prob_assoc"] = object_gb.apply(compute_proba, axis=1)
    # print(object_gb[["i:objectId", "assoc_grb", "grb_prob_assoc"]])
    # # print(agregated_objects[agregated_objects["i:objectId"] == "ZTF18abtpgto"])
