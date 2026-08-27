from joblib import Parallel, delayed
import pandas as pd

from tqdm import tqdm


def applyParallel(dfGrouped, func, n_jobs, print_progress, **kwargs):
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.Series(df_ls)
