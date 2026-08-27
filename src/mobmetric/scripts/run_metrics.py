import argparse
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt


import matplotlib.pyplot as plt
import powerlaw

from mobmetric import jump_length, location_frequency, radius_gyration, wait_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        default="count",
        nargs="?",
        choices=["duration", "count"],
        help="Method for calculating radius of gyration (default: %(default)s)",
    )
    parser.add_argument(
        "metric",
        default="jump",
        nargs="?",
        choices=["rg", "locf", "jump", "wait"],
        help="Metric to calculate (default: %(default)s)",
    )
    parser.add_argument(
        "dataset",
        default="dtepr_benchmark_1",
        nargs="?",
        help="Dataset for running (default: %(default)s)",
    )

    args = parser.parse_args()

    sp = pd.read_csv(os.path.join("data", "metrics", "input", f"{args.dataset}.csv"), index_col="index")
    sp["geometry"] = sp["geometry"].apply(wkt.loads)
    sp = gpd.GeoDataFrame(sp, geometry="geometry", crs="EPSG:4326")

    if args.metric == "jump":
        metric = jump_length(sp)
        xlabel = r"$\Delta r\,(m)$"
        ylabel = r"$P(\Delta r)$"
        xmin = 1

    elif args.metric == "rg":
        metric = radius_gyration(sp, method=args.method, print_progress=True)
        # transform to km
        metric = metric / 1000

        xlabel = "$Rg$ (km)"
        ylabel = "$P(Rg)$"
        xmin = 1

    elif args.metric == "wait":
        metric = wait_time(sp)

        xlabel = r"$\Delta t\,(hour)$"
        ylabel = r"$P(\Delta t)$"
        xmin = 0.1

    elif args.metric == "locf":
        loc_freq = location_frequency(sp)

        xlabel = "$f_k$"
        ylabel = "$k$"

    else:
        raise AttributeError(
            "Unsupported metric. Expected one of 'rg', 'locf', 'jump', or "
            f"'wait'; received {args.metric!r}."
        )

    plt.figure(figsize=(8, 5))

    if args.metric == "jump" or args.metric == "rg" or args.metric == "wait":
        # fit power law
        fit = powerlaw.Fit(metric, xmin=xmin)

        # plotting
        powerlaw.plot_pdf(metric[metric > xmin], label="data")
        # fit.power_law.plot_pdf(linestyle="--", label="powerlaw fit")
        # fit.truncated_power_law.plot_pdf(linestyle="--", label="truncated power law")
        # fit.lognormal.plot_pdf(linestyle="--", label="lognormal fit")
    else:
        n = np.arange(len(loc_freq)) + 1
        #plt.plot(n, np.power(n, -1.0) / 4, "--", label="$f_k\sim k^{-1}$", color="k")
        plt.plot(n, loc_freq)

        plt.yscale("log")
        plt.xscale("log")

    plt.legend(prop={"size": 13})
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    log_dir = os.path.join("outputs", "metrics")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, f"{args.metric}.png"), bbox_inches="tight", dpi=600)
