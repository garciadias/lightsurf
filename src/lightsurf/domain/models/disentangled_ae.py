"""Phase B: disentangled conditional autoencoder for abundance-free tagging.

Implements the de Mijolla, Ness, Viti & Wheeler (2021, arXiv:2103.06377)
idea in a minimal, CPU-testable form:

- encoder  ``z = enc(spectrum)`` — a CNN + LSTM stack (reuses the lightsurf
  recurrence), the *chemical* latent,
- decoder  ``x̂ = dec(z, Teff/logg)`` — reconstructs the spectrum given the
  latent *and* the known physical parameters,
- disentanglement head — a gradient-reversal layer makes ``z`` uninformative
  about Teff/logg, so the latent is free of non-chemical variation.

loss = MSE(spectrum, x̂)  +  λ · MSE(Teff/logg, û_from_reversed_latent)

``disentanglement_lambda = 0`` gives the plain conditional autoencoder
(no adversarial term); ``> 0`` enforces independence of ``z`` from Teff/logg.

Skeleton caveat: builds and trains on synthetic data (CPU smoke test); the
disentanglement *quality* must be validated on the GPU with real spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk


@tf.custom_gradient
def _flip_gradient(x: tf.Tensor, scale: float) -> tf.Tensor:
    def grad(dy: tf.Tensor) -> tuple[tf.Tensor, None]:
        return -scale * dy, None

    return x, grad


class GradientReversal(tfk.layers.Layer):
    """Reverses gradients (scale multiplies them) — the adversarial term."""

    def __init__(self, scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return _flip_gradient(x, self.scale)


@dataclass
class DisentangledSpectralAE:
    latent_dim: int = 16
    n_conditions: int = 2  # Teff, logg
    disentanglement_lambda: float = 0.1
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    verbose: int = 1
    checkpoint_path: str = "models/disentangled_ae.keras"

    def _build(self, input_dim: int) -> tfk.Model:
        spectrum = tfk.layers.Input((1, input_dim), name="spectrum")
        conditions = tfk.layers.Input((self.n_conditions,), name="conditions")

        # encoder: CNN + LSTM -> chemical latent z
        x = tfk.layers.Conv1D(32, 3, strides=2, padding="same")(spectrum)
        x = tfk.layers.PReLU()(x)
        x = tfk.layers.Conv1D(16, 3, strides=2, padding="same")(x)
        x = tfk.layers.PReLU()(x)
        x = tfk.layers.LSTM(self.latent_dim)(x)
        z = tfk.layers.Dense(self.latent_dim, name="latent")(x)

        # decoder: z + conditions -> reconstructed spectrum
        dec_in = tfk.layers.Concatenate()([z, conditions])
        y = tfk.layers.Dense(64, activation="relu")(dec_in)
        y = tfk.layers.Dense(128, activation="relu")(y)
        y = tfk.layers.Dense(input_dim)(y)
        y = tfk.layers.Reshape((1, input_dim), name="reconstruction")(y)

        # disentanglement head: reversed-latent -> predict conditions
        reversed_z = GradientReversal(scale=self.disentanglement_lambda)(z)
        u_hat = tfk.layers.Dense(16, activation="relu")(reversed_z)
        u_hat = tfk.layers.Dense(self.n_conditions, name="u_prediction")(u_hat)

        model = tfk.models.Model([spectrum, conditions], [y, u_hat])
        self.encoder = tfk.models.Model(spectrum, z, name="spectral_encoder")
        self.model = model
        return model

    def fit(
        self, X: np.ndarray, U: np.ndarray,
        validation_split: float = 0.2,
    ) -> tfk.callbacks.History:
        """X: (n, input_dim) spectra; U: (n, n_conditions) physical params."""
        input_dim = X.shape[1]
        self._build(input_dim)
        self.model.compile(
            optimizer=tfk.optimizers.Adam(self.learning_rate),
            loss={"reconstruction": "mse", "u_prediction": "mse"},
            loss_weights={"reconstruction": 1.0, "u_prediction": 1.0},
        )
        callbacks = [
            tfk.callbacks.EarlyStopping(monitor="val_loss", patience=5),
            tfk.callbacks.ModelCheckpoint(
                self.checkpoint_path, save_best_only=True,
            ),
        ]
        return self.model.fit(
            [X.reshape(X.shape[0], 1, input_dim).astype("float32"),
             U.astype("float32")],
            [X.reshape(X.shape[0], 1, input_dim).astype("float32"),
             U.astype("float32")],
            epochs=self.epochs, batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=self.verbose, callbacks=callbacks,
        )

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return the chemical latent z for each spectrum."""
        X = np.asarray(X, dtype="float32")
        if X.ndim < 3:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return self.encoder.predict(X)
