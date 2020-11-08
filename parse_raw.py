"""
 Created by Narayan Schuetz at 07.11.20 
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import datetime
import pandas as pd
import re
import numpy as np
import logging
from pyedflib import highlevel


def parse_raw_edf(file_path: str, participant_id: object, millisecond_timestamp=True) -> pd.DataFrame:
    """
    Parses new EMFIT-QS '.edf' file format.

    NOTE: The '.edf' files have a strange intrinsic start datetime. We've found it to be one hour ahead of the correct
          UTC time. In this script this is taken into account. However, it is strongly advised to double check the times
          as this could change since it seems like unintended behavior from Emfit's side. In addition, there is also
          some delay between the '.edf' version and the '.csv' files (including the non-raw ones) that can range from
          few seconds to more than 30 seconds - from what we've seen.

    :param file_path: path to raw emfit '.edf' file.
    :param participant_id: data source identifier.
    :param millisecond_timestamp: whether timestamp should be with millisecond precision or regular UNIX timestamp.
    :return: DataFrame containing upsampled high- and low-band EFMIT-QS data with the following format:

             Index:
                 RangeIndex
             Columns:
                 Name: participant_id, dtype: object
                 Name: presence_id, dtype: str
                 Name: record_timestamp, dtype: int64
                 Name: data_lowband, dtype: float32
                 Name: data_highband, dtype: float32
    """

    signals, signal_headers, header = highlevel.read_edf(file_path)
    emfit_raw_edf = pd.DataFrame({
        "participant_id": participant_id,
        "presence_id": "None",
        "record_timestamp": pd.date_range(
            start=datetime.datetime.utcfromtimestamp(header["startdate"].timestamp()) - datetime.timedelta(hours=1),
            periods=signals.shape[1],
            freq='{}N'.format(int(1e9 / 100))),
        "data_highband": signals[0].astype(np.float32),
        "data_lowband": signals[1].astype(np.float32)
    })
    emfit_raw_edf.record_timestamp = emfit_raw_edf.record_timestamp.apply(lambda x: int(x.timestamp() * 1000))

    if not millisecond_timestamp:
        emfit_raw_edf.record_timestamp /= 1000
    emfit_raw_edf.record_timestamp = emfit_raw_edf.record_timestamp.astype(np.int64)

    return emfit_raw_edf


def parse_raw_csv(file_path: str, participant_id: object, millisecond_timestamp=True) -> pd.DataFrame:
    """
    Parses new EMFIT-QS '.csv' file format, where the data sequences are properly quoted -> ("[...]").

    :param file_path: path to raw emfit '.csv' file.
    :param participant_id: data source identifier.
    :param millisecond_timestamp: whether timestamp should be with millisecond precision or regular UNIX timestamp.
    :return: DataFrame containing upsampled high- and low-band EFMIT-QS data with the following format:

             Index:
                 RangeIndex
             Columns:
                 Name: participant_id, dtype: object
                 Name: presence_id, dtype: str
                 Name: record_timestamp, dtype: int64
                 Name: data_lowband, dtype: float32
                 Name: data_highband, dtype: float32
    """
    raw_data = pd.read_csv(
        file_path,
        quotechar='"',
        header=0,
        usecols=list(range(5)),
        engine="c")

    _id = raw_data["id"].iloc[0]
    device_id = raw_data["device_serial"]
    timestamp_start = int(raw_data["start_date"].iloc[0])

    lowband = raw_data["data_lo_band"].iloc[0]
    lowband = np.array([float(e) for e in lowband[1:-1].split(",")], dtype=np.float32)

    highband = raw_data["data_hi_band"].iloc[0]
    highband = np.array([float(e) for e in highband[1:-1].split(",")], dtype=np.float32)

    n = len(highband)
    lowband = _x2_upsample_w_nan(lowband, n)

    if not n == len(lowband):
        raise ValueError("Error with upsampling, expected both signals to have the same nubmer of measurements. "
                         "Found number lowband = %d, number highband = %d!" % (len(lowband), len(highband)))

    millisecond_ts = _generate_millisecond_timestamps(timestamp_start, 100, n)

    df = pd.DataFrame({
        "participant_id": [participant_id] * n,
        "presence_id": ["None"] * n,
        "record_timestamp": millisecond_ts if millisecond_timestamp else millisecond_ts // 1000,
        "data_lowband": lowband,
        "data_highband": highband
    }, columns=[
        "participant_id",
        "presence_id",
        "record_timestamp",
        "data_lowband",
        "data_highband"
    ])

    return df.interpolate()


def parse_raw_csv_old_format(file_path: str, participant_id: object, millisecond_timestamp=True) -> pd.DataFrame:
    """
    Parses old EMFIT-QS '.csv' file format, where the data sequences are not quoted -> ([...]).

    :param file_path: path to raw emfit '.csv' file.
    :param participant_id: data source identifier.
    :param millisecond_timestamp: whether timestamp should be with millisecond precision or regular UNIX timestamp.
    :return: DataFrame containing upsampled high- and low-band EFMIT-QS data with the following format:

             Index:
                 RangeIndex
             Columns:
                 Name: participant_id, dtype: object
                 Name: presence_id, dtype: str
                 Name: record_timestamp, dtype: int64
                 Name: data_lowband, dtype: float32
                 Name: data_highband, dtype: float32
    """

    # NOTE: this is quite interesting, since we have nested quantifiers in our regexp pattern, it is possible to go
    # into catastrophic backtracking (https://www.regular-expressions.info/catastrophic.html) trying out almost
    # infinite amount of permutations. In essence this means certain corrupted files pose a huge problem if
    # they don't have the expected enf of file format. https://www.regular-expressions.info/catastrophic.html
    # the current workaround is to check whether the end of a raw file is correctly formatted.

    pattern = re.compile(r"(.+?),\s?(.+?),\s?(.+?),\s?(.+?),\s?\[(.*?)\],\s?\[(.*?)\],(.+)")
    corruption_check_pattern = re.compile(r",\d{9,10}$")
    logging.debug("extracting raw file %s" % file_path)

    with open(file_path, "r") as f:
        raw_content = f.read()

        # naive check for file without correct ending.
        if not corruption_check_pattern.search(raw_content[-100:]):
            raise IOError("Corrupt raw file '%s'.\n File ends with %s" % (file_path, raw_content[-15:]))

        matched = pattern.search(raw_content)

        if matched is None:
            if "error" in raw_content:  # there is no content
                return pd.DataFrame()
            else:  # could be anything, maybe a change in format
                raise IOError("file '%s' has no matching pattern, maybe a new format was introduced" % file_path)

        logging.debug("start extraction")

        _id = matched.group(1)
        device_id = matched.group(2)
        presence_id = matched.group(3)
        start_date_str = matched.group(4)  # might be EET/EETS timezone, so don't rely on that
        data_lowband_raw = matched.group(5)
        data_highband_raw = matched.group(6)
        remainder = matched.group(7)

        logging.debug("start extracting bands")

        data_lowband = np.array([float(record.strip()) for record in data_lowband_raw.split(",") if record != ""],
                                dtype=np.float32)
        if len(data_lowband) == 0:
            raise ValueError("No lowband records found!")

        data_highband = np.array([float(record.strip()) for record in data_highband_raw.split(",") if record != ""],
                                 dtype=np.float32)

        if len(data_highband) == 0:
            raise ValueError("No highband records found!")

        n = len(data_highband)

        logging.debug("start upsampling")
        data_lowband = _x2_upsample_w_nan(data_lowband, n)
        logging.debug("finished upsampling")
        timestamp_start = int(remainder.split(",")[-1])

        if not n == len(data_lowband):
            raise ValueError("Error with upsampling, expected both signals to have the same nubmer of measurements. "
                             "Found number lowband = %d, number highband = %d!" % (
                             len(data_lowband), len(data_highband)))

        millisecond_ts = _generate_millisecond_timestamps(timestamp_start, 100, n)

        df = pd.DataFrame({
            "participant_id": [participant_id] * n,
            "presence_id": [presence_id] * n,
            "record_timestamp": millisecond_ts if millisecond_timestamp else millisecond_ts // 1000,
            "data_lowband": data_lowband,
            "data_highband": data_highband
        }, columns=[
            "participant_id",
            "presence_id",
            "record_timestamp",
            "data_lowband",
            "data_highband"
        ])

        return df.interpolate()


def _x2_upsample_w_nan(array_like, n):
    return np.array([array_like[i // 2] if i % 2 == 0 else np.nan for i in range(n)], dtype=np.float32)


def _generate_millisecond_timestamps(start_timestamp_s, frequecy_hz, n):
    ts_ms = start_timestamp_s * 1000
    step = 1000 / frequecy_hz
    timestamps_ms = []
    for i in range(n):
        timestamps_ms.append(int(ts_ms))
        ts_ms += step
    return np.array(timestamps_ms)