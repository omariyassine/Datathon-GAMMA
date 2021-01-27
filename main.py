import pandas as pd
import numpy as np
from fbprophet import Prophet
import os
import sys

from config.own import (
    START_FORECAST,
    END_FORECAST,
    PARAMS,
)

from model import ProphetModel


def main(start=START_FORECAST, end=END_FORECAST, params=PARAMS):
    """
    Réalise la prédiction pour chaque arc sur la période souhaitée et avec les paramètres spécifiés.
    """
    dataframes = {arc: pd.read_csv(f"data/processed_{arc}.csv") for arc in ["champs", "conv", "sts"]}

    output = pd.DataFrame()

    for arc in ["champs", "conv", "sts"]:
        results = pd.DataFrame({"Arc": arc, "Datetime": pd.date_range(start, end, freq="H")})

        for target in ["debit", "taux"]:
            print(f"\nForecasting '{target}' for '{arc}'")
            param = params[arc][target]
            model = ProphetModel(param["training_start"], start, end, param["use_ext"], param["seasonality"])
            model.prepare(dataframes[arc], target)
            model.fit()
            y_pred = model.predict()
            results[target] = y_pred.reset_index()[target]
        
        output = pd.concat([output, results], axis=0)

    output.rename({"debit": "Débit horaire", "taux": "Taux d'occupation"}, axis=1, inplace=True)
    output.replace({"champs": "Champs-Elysées", "conv": "Convention", "sts": "Saints-Pères"}, inplace=True)

    if not os.path.exists("results"):
        os.makedirs("results")
    
    output.to_csv("results/prediction.csv", index=False, encoding="utf-8")

    return 0

if __name__ == "__main__":
    sys.exit(main())