from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.layers import LSTM, Bidirectional, Dense  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore


class LSTMRegressor:
    def __init__(
        self,
        checkpoint_path,
        lstm_units: int = 250,
        dense_units: int = 128,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        random_state: int = 42,
    ):
        self.checkpoint_path = checkpoint_path
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.callbacks = [
            EarlyStopping(monitor="val_loss", patience=5),
            ModelCheckpoint(
                checkpoint_path, save_best_only=True, save_weights_only=False
            ),
        ]
        self._build_model()
        self._compile_model()

    def __repr__(self):
        string_components = [
            f"LSTMRegressor(lstm_units={self.lstm_units},",
            f"dense_units={self.dense_units},",
            f"dropout={self.dropout},",
            f"epochs={self.epochs}, batch_size={self.batch_size},",
            f"verbose={self.verbose}, random_state={self.random_state})",
        ]
        return " ".join(string_components)

    def _build_model(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(self.lstm_units, dropout=self.dropout))),
        self.model.add(Dense(self.dense_units, activation="relu"))
        self.model.add(Dense(1))

    def _compile_model(self):
        self.model.compile(loss="mse", optimizer="adam", metrics=["mse"])

    def get_sequence_data(
        self,
        X: DataFrame,
        y: Series,
        sequence_split_by: Union[str, List[str]],
        n_steps: int = 3,
        group_by: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data to input a LSTM model.

        This method will group the data by the group_by columns and sort the data by the
        sequence_split_by columns. After that, it will split the data into sequences of
        of n_steps length.

        Parameters:
        ----------
        X: DataFrame
            The features data.
        y: Series
            The target data.
        group_by: List[str]
            The columns to group the data by.
        sequence_split_by: Union[str, List[str]]
            The columns to sort the and sequence the data by.
        n_steps: int
            The number of steps in the sequence.

        Returns:
        --------
        ndarray[M, n_steps, n_features], ndarray[M]
            The sequence data and the target data. M is the number of examples, n_steps
            is the number of steps in the sequence, and n_features is the number of
            features.
        """
        data = pd.concat([X, y], axis=1)
        data = data.sort_values(sequence_split_by)
        X_sequence_data = []
        y_target_data = []
        if group_by is None:
            for i in range(len(data) - n_steps):
                X_sequence_data.append(data.iloc[i:i + n_steps][X.columns].values)
                import pdb

                pdb.set_trace()  # noqa
                y_target_data.append(data.iloc[i + n_steps][str(y.name)])
            return np.array(X_sequence_data), np.array(y_target_data)
        for _, group in data.groupby(group_by):
            group = group.sort_values(sequence_split_by)
            for i in range(len(group) - n_steps):
                X_sequence_data.append(group.iloc[i:i + n_steps][X.columns].values)
                y_target_data.append(group.iloc[i + n_steps][str(y.name)])
        return np.array(X_sequence_data), np.array(y_target_data)

    def fit(
        self,
        X_train: DataFrame,
        y_train: Series,
        sequence_split_by: Optional[Union[str, List[str]]] = None,
        val_size: float = 0.2,
    ):
        print(self)
        x_train_, x_val, y_train_, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=self.random_state
        )
        if sequence_split_by is not None:
            x_train_, y_train_ = self.get_sequence_data(
                X_train, y_train, sequence_split_by
            )
        else:
            x_train_ = x_train_.values.reshape(x_train_.shape[0], 1, x_train_.shape[1])
            x_val = x_val.values.reshape(x_val.shape[0], 1, x_val.shape[1])
            y_train_, y_val = y_train_.values, y_val.values
        self.model.summary()
        self.model.fit(
            x_train_.astype("float32"),
            y_train_.astype("float32"),
            validation_data=(x_val.astype("float32"), y_val.astype("float32")),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )

    def predict(self, X: Union[DataFrame, np.ndarray[Any, Any]]):
        if isinstance(X, DataFrame):
            X = X.values.reshape(X.shape[0], 1, X.shape[1]).astype("float32")
        return self.model.predict(X.astype("float32")).flatten()

    def get_params(self, deep: bool = False):
        return {
            "checkpoint_path": self.checkpoint_path,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(
        self,
        X: Union[DataFrame, np.ndarray[Any, Any]],
        y: Union[ExtensionArray, Series, np.ndarray[Any, Any]],
    ):
        # Return the validation loss
        if isinstance(X, DataFrame):
            X = X.values.reshape(X.shape[0], 1, X.shape[1]).astype("float32")
        if isinstance(y, Series):
            y = y.values.astype("float32")
        results = self.model.evaluate(X.astype("float32"), y.astype("float32"))
        self.model.summary()
        print(
            f"After {self.epochs} epochs:\n"
            "MSE:\n"
            f"Train: {results[0]:0.3f}, Validation: {results[1]:0.3f}"
        )
        return results[1]
