import pandas as pd
import numpy as np
from fbprophet import Prophet


class ProphetModel():

    """ Prédicteur basé sur le modèle Prophet de Facebook."""

    def __init__(self, start_train, start_pred, end_pred, use_ext, seasonality):
        """
        Arg(s):
         - start_train: <pandas.Timestamp> date de début d'entraînement;        
         - start_pred: <pandas.Timestamp> date de début de prédiction;
         - end_pred: <pandas.Timestamp> date de fin de prédiction;
         - use_ext: <bool> utiliser les variables exogènes ou non;
         - seasonality: <str> type de saisonnalité à utiliser ("additive" ou "multiplicative").
        """
        self.start_train = start_train
        self.start_pred = start_pred 
        self.end_pred = end_pred
        self.use_ext = use_ext
        self.prophet = Prophet(seasonality_mode=seasonality)
    

    def prepare(self, data, target, date_column="date_heure"):
        """
        Prépare les données pour le modèle.
        -------
        Arg(s):
         - data: <pandas.DataFrame> données de l'arc à prédire;
         - target: <str> quantité à prédire;
         - date_column: <str> colonne contenant les dates.
        """
        self.target = target
        if not self.use_ext:
            self.df = data[[date_column, target]]
            self.df.date_heure = pd.to_datetime(self.df.date_heure)
            self.df.rename({"date_heure": "ds", target: "y"}, axis=1, inplace=True)
        else:
            self.df = data.drop(["Unnamed: 0", "date", "debit"*(target!="debit") + "taux"*(target!="taux")], axis=1)
            self.df.date_heure = pd.to_datetime(self.df.date_heure)
            self.df.rename({"date_heure": "ds", target: "y", "holidays": "holidays_"}, axis=1, inplace=True)
        self.df.fillna(0, inplace=True)
        self.df_train = self.df.loc[(self.df.ds>=self.start_train) & (self.df.ds<self.start_pred)]
        self.df_test = self.df.loc[(self.df.ds>=self.start_pred) & (self.df.ds<=self.end_pred)]
    

    def fit(self):
        if self.use_ext:
            for col in self.df_train.columns:
                if col not in ["ds", "y"]:
                    self.prophet.add_regressor(col)
        self.prophet.fit(self.df_train)
    
    
    def predict(self):
        future = self.df_test.drop("y", axis=1)
        self.forecast = self.prophet.predict(future)
        self.y_pred = self.forecast.loc[(self.forecast.ds>=self.start_pred) & (self.forecast.ds<=self.end_pred)][["ds", "yhat"]].rename({"yhat": self.target}, axis=1)
        self.y_pred.loc[self.y_pred[self.target]<0, self.target] = 0
        self.y_pred.set_index("ds", inplace=True)

        return self.y_pred

    