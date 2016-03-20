import pandas as pd
import csv
import time


def write_submission(y_pred, dataframe):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    s1 = pd.Series(dataframe.ID, name='ID').reset_index(drop=True)
    s2 = pd.Series(y_pred, name='TARGET')
    results = pd.concat([s1, s2], axis=1)
    results = results.sort_values('ID')
    results.to_csv(path_or_buf='generated/submissions/predictions_%s.csv' % timestr, index=False, quoting=csv.QUOTE_NONNUMERIC)
