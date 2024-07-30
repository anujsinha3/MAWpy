import logging

import numpy as np
import pandas as pd

from mawpy.constants import (USER_ID, UNIX_START_DATE, ORIG_LAT, ORIG_LONG, UNIX_START_T,
                             STAY_LAT, STAY_LONG, STAY_DUR, STAY, TSC_COLUMNS)
from mawpy.distance import distance
from mawpy.utilities.common import get_combined_stay, get_stay_groups
from mawpy.utilities.preprocessing import get_preprocessed_dataframe, get_list_of_chunks_by_column, execute_parallel

logger = logging.getLogger(__name__)


def _does_diameter_constraint_exceed(starting_index: int, ending_index: int,
                                     latitudes_list: list[float], longitudes_list: list[float],
                                     spatial_constraint: float, distance_map: dict) -> bool:
    """
    Check if the distance between any two points within a given range exceeds the spatial constraint.

    Args:
        starting_index (int): The starting index of the range.
        ending_index (int): The ending index of the range.
        latitudes_list (list[float]): List of latitudes.
        longitudes_list (list[float]): List of longitudes.
        spatial_constraint (float): The maximum allowed distance.
        distance_map (dict): A dictionary to cache distances between points.

    Returns:
        bool: True if any distance exceeds the spatial constraint, False otherwise.
    """

    for i in range(ending_index, starting_index - 1, -1):
        for j in range(i - 1, starting_index - 1, -1):

            point_key_combo_1 = (latitudes_list[i], longitudes_list[i], latitudes_list[j], longitudes_list[j])
            point_key_combo_2 = (latitudes_list[j], longitudes_list[j], latitudes_list[i], longitudes_list[i])

            if point_key_combo_1 not in distance_map:
                if point_key_combo_2 not in distance_map:
                    calculated_distance = distance(point_key_combo_1[0], point_key_combo_1[1],
                                                   point_key_combo_1[2], point_key_combo_1[3])
                    distance_map[point_key_combo_1] = calculated_distance
                    distance_map[point_key_combo_2] = calculated_distance
                else:
                    calculated_distance = distance_map[point_key_combo_2]

            else:
                calculated_distance = distance_map[point_key_combo_1]

            if calculated_distance > spatial_constraint:
                return True

    return False


def _does_duration_threshold_exceed(point_i: int, point_j: int, timestamps_list: list, duration_constraint: float) \
        -> bool:
    """
    Check if the time difference between two points exceeds the duration constraint.

    Args:
        point_i (int): The index of the first point.
        point_j (int): The index of the second point.
        timestamps_list (list[float]): List of timestamps.
        duration_constraint (float): The maximum allowed time difference.

    Returns:
        bool: True if the time difference exceeds the duration constraint, False otherwise.
    """
    return timestamps_list[point_j] - timestamps_list[point_i] >= duration_constraint


def _get_df_with_stays(each_day_df: pd.DataFrame, spatial_constraint: float, dur_constraint: float) -> pd.DataFrame:
    """
        For the trace of a user on a given day, the function calculates and assigns stay_lat and stay_long
        to each of the daily trace.

        It groups together traces into a 'stay' for which time difference between
            all the calculated diameter is within the spatial constraint and
            the first and the last point exceeds duration_threshold
    Args:
        each_day_df (pd.DataFrame): DataFrame containing traces for a single day.
        spatial_constraint (float): The maximum allowed distance between points in a stay.
        dur_constraint (float): The minimum required duration for a stay.

    Returns:
        pd.DataFrame: DataFrame with stay latitude, longitude, and duration added.

    """

    latitudes_for_day = each_day_df[ORIG_LAT].to_numpy()
    longitudes_for_day = each_day_df[ORIG_LONG].to_numpy()
    timestamps_for_day = each_day_df[UNIX_START_T].to_numpy()
    number_of_traces_for_day = len(each_day_df)

    stay_lat = np.full(number_of_traces_for_day, -1.0)
    stay_long = np.full(number_of_traces_for_day, -1.0)
    stay_dur = np.zeros(number_of_traces_for_day)

    start = 0
    distance_map = {}
    while start < number_of_traces_for_day:

        j = start
        while j < number_of_traces_for_day and not _does_duration_threshold_exceed(start, j, timestamps_for_day,
                                                                                   dur_constraint):
            j += 1

        if j == number_of_traces_for_day:
            break
        else:
            if _does_diameter_constraint_exceed(start, j, latitudes_for_day, longitudes_for_day, spatial_constraint,
                                                distance_map):
                start += 1
            else:
                cluster_end_index = j
                for k in range(j + 1, number_of_traces_for_day):
                    if not _does_diameter_constraint_exceed(start, k, latitudes_for_day, longitudes_for_day,
                                                            spatial_constraint, distance_map):
                        cluster_end_index = k
                    else:
                        break
                stay_lat[start: cluster_end_index + 1] = np.mean(latitudes_for_day[start: cluster_end_index + 1])
                stay_long[start: cluster_end_index + 1] = np.mean(longitudes_for_day[start: cluster_end_index + 1])
                stay_dur[start: cluster_end_index + 1] = timestamps_for_day[cluster_end_index] - timestamps_for_day[
                    start]

                start = cluster_end_index + 1

    each_day_df[STAY_LAT] = stay_lat
    each_day_df[STAY_LONG] = stay_long
    each_day_df[STAY_DUR] = stay_dur

    return each_day_df


def _run_for_user(df_by_user: pd.DataFrame, spatial_constraint: float, dur_constraint: float) -> pd.DataFrame:
    """
        Process traces for a user, assigning stay information.

        Args:
            df_by_user (pd.DataFrame): DataFrame containing traces for a single user.
            spatial_constraint (float): The maximum allowed distance between points in a stay.
            dur_constraint (float): The minimum required duration for a stay.

        Returns:
            pd.DataFrame: DataFrame with stay information added.
    """

    df_with_stay = df_by_user.groupby(UNIX_START_DATE).apply(
        lambda x: _get_df_with_stays(x, spatial_constraint, dur_constraint))
    df_with_stay[STAY] = get_stay_groups(df_with_stay)

    df_with_stay_added = get_combined_stay(df_with_stay)

    return df_with_stay_added


def _run(df_by_user_chunk: pd.DataFrame, args: tuple) -> pd.DataFrame:
    """
        Apply trace segmentation and clustering to chunks of user data.

        Args:
            df_by_user_chunk (pd.DataFrame): DataFrame chunk for a single user.
            args (tuple): Tuple containing spatial and duration constraints.

        Returns:
            pd.DataFrame: DataFrame with trace segmentation clustering applied.
    """
    spatial_constraint, dur_constraint = args
    df_by_user_chunk = (df_by_user_chunk.groupby(USER_ID)
                        .apply(lambda x: _run_for_user(x, spatial_constraint, dur_constraint)))
    return df_by_user_chunk


def trace_segmentation_clustering(output_file: str, spatial_constraint: float, dur_constraint: float,
                                  input_df: pd.DataFrame | None = None, input_file: str = None) -> pd.DataFrame | None:
    """
        Perform trace segmentation clustering, and save the results to a file.

        Args:
            output_file (str): Path to the output file.
            spatial_constraint (float): The maximum allowed distance between points in a stay.
            dur_constraint (float): The minimum required duration for a stay.
            input_df (pd.DataFrame, optional): Input DataFrame. Defaults to None.
            input_file (str, optional): Path to the input file. Defaults to None.

        Returns:
            pd.DataFrame | None: DataFrame with trace segmentation
            clustering results, or None if input is invalid.
    """
    if input_df is None and input_file is None:
        logger.error("At least one of input file path or input dataframe is required")
        return None

    if input_df is None:
        input_df = get_preprocessed_dataframe(input_file)

    user_id_chunks = get_list_of_chunks_by_column(input_df, USER_ID)
    # input_df.set_index(keys=[USER_ID], inplace=True)
    args = (spatial_constraint, dur_constraint)
    output_df = execute_parallel(user_id_chunks, input_df, _run, args)

    output_columns = list(set(TSC_COLUMNS) & set(output_df.columns))
    output_df = output_df[output_columns]
    output_df.dropna(how="all")
    output_df.to_csv(output_file, index=False)
    return output_df
