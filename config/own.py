import pandas as pd


START_FORECAST = pd.Timestamp("2020-12-11 01:00:00") # date de début de prédiction
END_FORECAST = pd.Timestamp("2020-12-16 23:00:00") # date de fin de prédiction

PARAMS = { # paramètres à utiliser pour chaque time series
    "champs": {
        "debit": {
            "use_ext": True,
            "seasonality": "additive",
            "training_start": pd.Timestamp("2020-09-01 00:00:00")
        },
        "taux": {
            "use_ext": True,
            "seasonality": "multiplicative",
            "training_start": pd.Timestamp("2020-01-01 00:00:00")
        }
    },
    "conv": {
        "debit": {
            "use_ext": True,
            "seasonality": "additive",
            "training_start": pd.Timestamp("2020-03-01 00:00:00")
        },
        "taux": {
            "use_ext": False,
            "seasonality": "multiplicative",
            "training_start": pd.Timestamp("2020-05-01 00:00:00")
        }
    },
    "sts": {
        "debit": {
            "use_ext": True,
            "seasonality": "additive",
            "training_start": pd.Timestamp("2020-03-01 00:00:00")
        },
        "taux": {
            "use_ext": True,
            "seasonality": "multiplicative",
            "training_start": pd.Timestamp("2020-03-01 00:00:00")
        }
    }
}