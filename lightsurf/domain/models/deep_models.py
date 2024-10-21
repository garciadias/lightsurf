from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.api.extensions import ExtensionArray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk


class AttentionLayer(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create the trainable weight variable for this layer.
        self.attention_weights = self.add_weight(shape=(input_shape[-1], 1),
                                                 initializer='glorot_uniform',
                                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Apply dense layer (tanh) to get attention scores
        attention_scores = tfk.activations.tanh(tf.matmul(x, self.attention_weights))
        # Remove last dimension
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        # Apply softmax to get attention weights
        attention_weights = tfk.activations.softmax(attention_scores)
        # Multiply input by attention weights
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        weighted_output = x * attention_weights
        # Sum the weighted sequence
        output = tf.reduce_sum(weighted_output, axis=1)
        return output


@dataclass
class CnnLstmAttentionModel():
    epochs: int = 100
    batch_size: int = 64
    loss: Literal["mse", "mae"] = "mse"
    loss_metrics: List[str] = field(default_factory=lambda: ["mae"])
    validation_split: float = 0.2
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    verbose: int = 1
    cnn_kernel_size: int = 3
    cnn_strides: int = 2
    cnn_filters: List[int] | int = field(default_factory=lambda: [999, 499, 249, 124, 61])
    lstm_units: List[int] | int = 256
    dense_units: List[int] | int = field(default_factory=lambda: [20, 8])
    dense_activation: str = "relu"
    early_stopping_patience: int = 5
    early_stopping_monitor: str = "val_loss"
    checkpoint_path: str | Path = "models/cnn_lstm_attention_model.keras"
    save_best_only: bool = True
    save_weights_only: bool = False

    def __post_init__(self):
        self.callbacks = [
            tfk.callbacks.EarlyStopping(
                monitor=self.early_stopping_monitor,
                patience=self.early_stopping_patience,
            ),
            tfk.callbacks.ModelCheckpoint(
                self.checkpoint_path,
                save_best_only=self.save_best_only,
                save_weights_only=self.save_weights_only,
            ),
        ]

    def __repr__(self):
        model_name = "CnnLstmAttentionModel"
        parameters = ", ".join(
            [
                f"{key}={value}"
                for key, value in self.__dict__.items()
                if not key.startswith("_")
            ]
        )
        return f"{model_name}({parameters})"

    def create_cnn_lstm_am(self, input_shape) -> tfk.Model:
        # Input layer
        inputs = tfk.layers.Input(shape=input_shape)

        if isinstance(self.dense_units, int):
            dense_units = [self.dense_units]
        else:
            dense_units = self.dense_units
        if isinstance(self.lstm_units, int):
            lstm_units = [self.lstm_units]
        else:
            lstm_units = self.lstm_units
        if isinstance(self.cnn_filters, int):
            cnn_filters = [self.cnn_filters]
        else:
            cnn_filters = self.cnn_filters
        # CNN layers

        for i, filter_dim in enumerate(cnn_filters):
            if i == 0:
                x = tfk.layers.Conv1D(
                    filters=filter_dim,
                    kernel_size=self.cnn_kernel_size,
                    strides=self.cnn_strides,
                    padding='same'
                )(inputs)
                x = tfk.layers.BatchNormalization()(x)
                x = tfk.layers.PReLU()(x)
            else:
                x = tfk.layers.Conv1D(
                    filters=filter_dim,
                    kernel_size=self.cnn_kernel_size,
                    strides=self.cnn_strides,
                    padding='same'
                )(x)
                x = tfk.layers.BatchNormalization()(x)
                x = tfk.layers.PReLU()(x)

        # LSTM layers
        for i, lstm_unit in enumerate(lstm_units):
            x = tfk.layers.LSTM(lstm_unit, return_sequences=True)(x)

        # Attention mechanism
        attention_output = AttentionLayer()(x)

        # Fully connected layers
        for i, dense_unit in enumerate(dense_units):
            if i == 0:
                x = tfk.layers.Dense(
                    dense_unit, activation=self.dense_activation
                )(attention_output)
            else:
                x = tfk.layers.Dense(
                    dense_unit, activation=self.dense_activation
                )(x)

        # Output layer for regression task
        output = tfk.layers.Dense(1)(x)

        # Define the model
        model = tfk.models.Model(inputs, output, name="cnn_lstm_attention_model")

        return model

    def _compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _create_optimizer(self):
        return tfk.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
        )

    def fit(self, X_train, y_train) -> tfk.callbacks.History:
        n_stars = X_train.shape[0]
        n_features = X_train.shape[1]
        train_data = X_train.values.reshape(n_stars, 1, n_features).astype('float32')
        train_labels = y_train.values.astype('float32')
        self.model = self.create_cnn_lstm_am((1, n_features))
        optimizer = self._create_optimizer()
        self._compile(optimizer=optimizer, loss=self.loss, metrics=self.loss_metrics)
        self.model.summary()
        history = self.model.fit(
            train_data,
            train_labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )
        return history

    def summary(self):
        self.model.summary()

    def predict(self, y_test):
        return self.model.predict(
            y_test.values.reshape(
                y_test.shape[0],
                1,
                y_test.shape[1]).astype('float32')
        ).flatten()

    def score(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }


@dataclass
class LSTMRegressor:
    checkpoint_path: str | Path
    output_dimension: int = 1
    loss: Literal["mse", "mae"] = "mse"
    lstm_units: int = 250
    dense_units: int = 128
    learning_rate: float = 0.001
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 32
    verbose: int = 1
    random_state: int = 42
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07

    def __post_init__(self):
        raise NotImplementedError(
            "This class is not implemented yet. There are a few bugs in the code."
        )
        self.callbacks = [
            tfk.callbacks.EarlyStopping(monitor="val_loss", patience=5),
            tfk.callbacks.ModelCheckpoint(
                self.checkpoint_path, save_best_only=True, save_weights_only=False
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
        lstm_layer = tfk.layers.LSTM(
            self.lstm_units, dropout=self.dropout, return_sequences=True
        )
        dense_layer = tfk.layers.Dense(self.dense_units, activation="relu")
        output_layer = tfk.layers.Dense(self.output_dimension)

        self.model = tfk.models.Sequential()
        self.model.add(lstm_layer),
        self.model.add(dense_layer)
        self.model.add(output_layer)

    def _compile_model(self):
        optmizer = tfk.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
        )
        self.model.compile(
            loss=self.loss,
            optimizer=optmizer,
            metrics=[self.loss],
        )

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
