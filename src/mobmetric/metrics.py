import numpy as np
from tqdm import tqdm
import pandas as pd

from trackintel.geogr.point_distances import haversine_dist


def radius_gyration(sp, print_progress=False, method="count"):
    """
    Radius of gyration for individuals.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with column "user_id" and geometry.

    print_progress: boolen, default False
        Show per-user progress if set to True.

    method: string, {"duration", "count"}, default "count"
        method to calculate rg. Duration additionally weights each sp with the activity duration.

    Returns
    -------
    pandas Series
        the radius of gyration for individuals.

    References
    ----------
    [1] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008). Understanding individual human mobility patterns. Nature, 453(7196), 779-782.

    """
    if print_progress:
        tqdm.pandas(desc="User radius of gyration calculation")
        s = sp.groupby("user_id").progress_apply(lambda x: _radius_gyration_user(x, method))
    else:
        s = sp.groupby("user_id").apply(lambda x: _radius_gyration_user(x, method))

    s.rename("radiusGyration", inplace=True)
    return s


def jump_length(sp):
    """
    Jump length between consecutive locations.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with geometry in latitude and longitude.

    Returns
    -------
    np.array
        Array containing the jump lengths.

    References
    ----------
    [1] Brockmann, D., Hufnagel, L., & Geisel, T. (2006). The scaling laws of human travel. Nature, 439(7075), 462-465.

    """
    pts = sp.geometry.values
    return np.array([haversine_dist(pts[i - 1].x, pts[i - 1].y, pts[i].x, pts[i].y)[0] for i in range(1, len(pts))])


def wait_time(df):
    """
    Wait time consecutive locations.

    Parameters
    ----------
    sp : DataFrame
        Staypoints with time information, either provided in "duration" column, or in "finished_at" and "started_at" columns.

    Returns
    -------
    np.array
        Array containing the wait time.

    References
    ----------
    [1] Brockmann, D., Hufnagel, L., & Geisel, T. (2006). The scaling laws of human travel. Nature, 439(7075), 462-465.

    """
    if "duration" in df.columns:
        return df["duration"].values
    else:
        # TODO: check
        return ((df["finished_at"] - df["started_at"]).dt.total_seconds() / 3600).values


def location_frequency(sp):
    """Calculate ranked location visit frequencies for a dataset.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with column "location_id" and "user_id".

    Returns
    -------
    s: list
        The ranked visit frequency.

    References
    ----------
    [1] Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008). Understanding individual human mobility patterns. Nature, 453(7196), 779-782.

    """

    # get visit times per user and location
    freq = sp.groupby(["user_id", "location_id"], as_index=False).size()
    # get the rank of locations per user
    freq["visitRank"] = freq.groupby("user_id")["size"].rank(ascending=False, method="first")
    # get the average visit freqency for every rank
    pLoc = freq.groupby("visitRank")["size"].mean().values

    # normalize
    pLoc = pLoc / pLoc.sum()

    return pLoc


# Backwards-compatible alias retained for the original project API.
location_frquency = location_frequency


def _radius_gyration_user(sp_user, method):
    """
    User level radius of gyration calculation, see radius_gyration() for details.

    Parameters
    ----------
    sp_user : Geodataframe
        The staypoints from an individual, should contain "geometry".

    method: string, {"duration", "count"}, default "count"
        method to calculate rg. Duration additionally weights each sp with the activity duration.
    Returns
    -------
    float
        the radius of gyration of the individual
    """
    sp_user["lat"] = sp_user["geometry"].y
    sp_user["lng"] = sp_user["geometry"].x
    lats_lngs = sp_user[["lat", "lng"]].values

    if method == "duration":
        durs = sp_user[["duration"]].values

        center_of_mass = np.sum([lat_lng * dur for lat_lng, dur in zip(lats_lngs, durs)], axis=0) / np.sum(durs)
        inside = [
            dur * (haversine_dist(lng, lat, center_of_mass[-1], center_of_mass[0]) ** 2.0)
            for (lat, lng), dur in zip(lats_lngs, durs)
        ]

        rg = np.sqrt(np.sum(inside) / sp_user["duration"].sum())
    elif method == "count":
        center_of_mass = np.mean(lats_lngs, axis=0)
        rg = np.sqrt(
            np.mean([haversine_dist(lng, lat, center_of_mass[-1], center_of_mass[0]) ** 2.0 for lat, lng in lats_lngs])
        )
    else:
        raise AttributeError(
            f"Method unknown. Please check the input arguement. We only support 'duration', 'count'. You passed {method}"
        )
    return rg
