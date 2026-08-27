import numpy as np
import pandas as pd

from networkx.algorithms import isomorphism
import networkx as nx

from trackintel.analysis.tracking_quality import _split_overlaps

from tqdm import tqdm


def mobility_motifs(sp, proportion_filter=0.005):
    """
    Get the mobility motifs for a input dataset (sp).

    Mobility motifs are defined as unqiue patterns of location visits per user day. Thus, location visits of each user are binned per day, and cross-compared to find valid mobility motifs. A mobility motif shall fulfil the following requirements:
    - Contains all locations visited by a user in a day.
    - Frequently occuring in the dataset (controlled by the parameter `proportion_filter`).

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with user and time information ("user_id", "started_at", "finished_at"), and "location_id".

    proportion_filter: boolen, default 0.005
        Filter to control how frequent a pattern coulf be considered a motifs, e.g., 0.005 means patterns occuring more than 0.5% of all the patterns are considered motifs.

    Returns
    -------
    pandas DataFrame
        User day dataframe containing the motifs information with columns "visits", "uniq_visits", and "class". "visits" and "uniq_visits" represent the number of locations and number of unique location visits during the day, repectively. "class" is the unique type of motifs of the day. "uniq_visits" and "class" together uniquely define a motif. Non motif days receive NaN value.

    References
    ----------
    [1] Schneider, C. M., Belik, V., Couronné, T., Smoreda, Z., & González, M. C. (2013). Unravelling daily human mobility motifs. Journal of The Royal Society Interface, 10(84), 20130246.

    """
    # split the records based on day, such that daily motifs can be constructed
    sp = _split_overlaps(sp, granularity="day")
    sp["date"] = sp["started_at"].dt.date

    # delete the self transitions within the same day (no required for generated sequences)
    sp["loc_next"] = sp["location_id"].shift(-1)
    sp["date_next"] = sp["date"].shift(-1)

    sp = sp.loc[~((sp["loc_next"] == sp["location_id"]) & (sp["date_next"] == sp["date"]))].copy()
    sp.drop(columns=["loc_next", "date_next"], inplace=True)

    # count unique daily location visits, and merge back to sp
    user_date_loc_count = sp.groupby(["user_id", "date"]).agg({"location_id": "nunique"})
    user_date_loc_count.rename(columns={"location_id": "uniq_visits"}, inplace=True)
    sp = sp.merge(user_date_loc_count.reset_index(), on=["user_id", "date"], how="left")

    # construct possible graphs
    user_day_df = _get_user_day_graph(sp)

    # get total number of graphs for filtering
    total_graphs = len(user_day_df)

    def _get_valid_motifs(df):
        if (len(df) / total_graphs) > proportion_filter:
            return df

    # get the valid motifs per user days
    motifs_user_days = (
        user_day_df.groupby(["uniq_visits", "class"], as_index=False).apply(_get_valid_motifs).reset_index(drop=True)
    )

    # merge back to all user days
    return_df = (
        sp.groupby(["user_id", "date"])
        .size()
        .rename("visits")
        .reset_index()
        .merge(motifs_user_days, on=["user_id", "date"], how="left")
    )

    return return_df


def _get_user_day_graph(sp):
    """
    Construct network patterns from user daily location visits. The return of the function can be used to filter for motifs.

    Parameters
    ----------
    sp : Geodataframe
        Staypoints with columns "user_id", "date", "uniq_visits" and "location_id".

    Returns
    -------
    pandas DataFrame
        User day dataframe containing the network pattern information with columns "uniq_visits", "class". "uniq_visits" represents the number of unique location visits during the day. "class" is the unique type of network pattern of the day. "uniq_visits" and "class" together uniquely define a pattern. Non pattern days receive NaN value.

    References
    ----------
    [1] Schneider, C. M., Belik, V., Couronné, T., Smoreda, Z., & González, M. C. (2013). Unravelling daily human mobility motifs. Journal of The Royal Society Interface, 10(84), 20130246.

    """

    user_day_ls = []

    # consider up to 6 location visits per day
    for uniq_visits in tqdm(range(1, 7)):
        curr_sp = sp.loc[sp["uniq_visits"] == uniq_visits].copy()
        curr_sp["next_loc"] = curr_sp["location_id"].shift(-1)

        # for only 1 location visit, every day is the same motif
        if uniq_visits == 1:
            graph_s = curr_sp.groupby(["user_id", "date"]).size().rename("class").reset_index()
            graph_s["class"] = 0
            graph_s["uniq_visits"] = uniq_visits
            user_day_ls.append(graph_s)
            continue

        # the edge number shall be at least the node number, otherwise not a motif
        edge_num = curr_sp.groupby(["user_id", "date"]).size() - 1
        valid_user_dates = edge_num[edge_num >= uniq_visits].rename("edge_num")
        curr_sp = curr_sp.merge(valid_user_dates.reset_index(), on=["user_id", "date"], how="left")
        curr_sp = curr_sp.loc[~curr_sp["edge_num"].isna()]

        # for 2 location visits, every day that have the edge number larger than the node number is the same motif
        if uniq_visits == 2:
            graph_s = curr_sp.groupby(["user_id", "date"]).size().rename("class").reset_index()
            graph_s["class"] = 0
            graph_s["uniq_visits"] = uniq_visits
            user_day_ls.append(graph_s)
            continue

        graph_s = curr_sp.groupby(["user_id", "date"]).apply(_construct_day_graph)
        # valid motifs shall be connected: each node shall have in and our degree
        # filter graphs that do not have an in-degree and out degree
        graph_s = graph_s.loc[~graph_s.isna()]

        # check for motif groups
        unique_motif_group_ls = []
        for i in range(len(graph_s) - 1):
            # if i has already been check, we do not need to check again
            if i in [item for sublist in unique_motif_group_ls for item in sublist]:
                continue

            # check for repeated patterns as i in [i+1, max)
            possible_match_ls = [i]
            for j in range(i + 1, len(graph_s)):
                if isomorphism.GraphMatcher(graph_s.iloc[i], graph_s.iloc[j]).is_isomorphic():
                    possible_match_ls.append(j)

            # append this group of motif
            unique_motif_group_ls.append(possible_match_ls)

        # label motif class and assign back to graph_s
        graph_s = graph_s.rename("graphs").reset_index()
        class_arr = np.zeros(len(graph_s))
        for i, classes in enumerate(unique_motif_group_ls):
            class_arr[classes] = i
        graph_s["class"] = class_arr
        graph_s["class"] = graph_s["class"].astype(int)
        graph_s["uniq_visits"] = uniq_visits
        graph_s.drop(columns={"graphs"}, inplace=True)

        user_day_ls.append(graph_s)

    return pd.concat(user_day_ls)


def _construct_day_graph(df):
    """
    Construct networks from daily location visits. Shall be used after groupby ["user_id", "date"].

    Parameters
    ----------
    df : Geodataframe
        Staypoints with columns "location_id" and "next_loc".

    Returns
    -------
    networkx DiGraph
        Graph object constructed from location visits.

    """
    G = nx.DiGraph()
    G.add_nodes_from(df["location_id"])

    G.add_edges_from(df.iloc[:-1][["location_id", "next_loc"]].astype(int).values)

    in_degree = np.all([False if degree == 0 else True for _, degree in G.in_degree])
    out_degree = np.all([False if degree == 0 else True for _, degree in G.out_degree])
    # TODO: check the requirement in the original paper
    if in_degree and out_degree:
        return G
