import sys
import pandas as pd

def get_mae(pred, gold):
    pred_df = pd.read_csv(pred, delimiter=",", quotechar='"')
    gold_df = pd.read_csv(gold, delimiter=",", quotechar='"')
    err_list = []

    for dv in ["nFix", "FFD", "GPT", "TRT", "fixProp"]:
        err = pred_df[dv] - gold_df[dv]
        err_list.append(err.abs().mean())
        # err_list.append((err**2).mean())
        print("MAE for {} is {}".format(dv, round(err.abs().mean(), 3)))
        # print("MSE for {} is {}".format(dv, round((err**2).mean(), 3)))
    print("Total MAE is {}".format(round(sum(err_list)/5, 4)))

get_mae(sys.argv[1], sys.argv[2])