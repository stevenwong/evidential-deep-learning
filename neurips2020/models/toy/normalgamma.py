import tensorflow as tf
import tensorflow_probability as tfp
import functools
import evidential_deep_learning as edl

def create(
    input_shape,
    num_neurons=24,
    num_layers=1,
    activation=tf.nn.relu,
    num_ensembles=5,
    ):

    options = locals().copy()

    # def create_model():
    #     inputs = tf.keras.Input(input_shape)
    #     x = inputs
    #     for _ in range(num_layers):
    #         x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    #     output = edl.layers.DenseNormalGamma(1)(x)
    #     model = tf.keras.Model(inputs=inputs, outputs=output)
    #     return model

    # num_neurons = int(num_neurons / 4)
    # def create_model():
    #     inputs = tf.keras.Input(input_shape)

    #     # gamma
    #     x = inputs
    #     for _ in range(num_layers):
    #         x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    #     mu = tf.keras.layers.Dense(1, activation='linear')(x)

    #     # nu
    #     x = inputs
    #     for _ in range(num_layers):
    #         x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    #     x = tf.keras.layers.Dense(1, activation='linear')(x)
    #     v = tf.nn.softplus(x)

    #     # alpha
    #     x = inputs
    #     for _ in range(num_layers):
    #         x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    #     x = tf.keras.layers.Dense(1, activation='linear')(x)
    #     alpha = tf.nn.softplus(x) + 1.

    #     # beta
    #     x = inputs
    #     for _ in range(num_layers):
    #         x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
    #     x = tf.keras.layers.Dense(1, activation='linear')(x)
    #     beta = tf.nn.softplus(x)

    #     model = tf.keras.Model(inputs=inputs, outputs=tf.concat([mu, v, alpha, beta], axis=-1))
    #     return model

    def create_model():
        inputs = tf.keras.Input(input_shape)

        # gamma
        x = inputs
        for _ in range(num_layers):
            x = tf.keras.layers.Dense(num_neurons, activation=activation)(x)
        latent = x

        x = tf.keras.layers.Dense(int(num_neurons/4), activation=activation)(x)
        mu = tf.keras.layers.Dense(1, activation='linear')(x)

        # nu
        x = latent
        x = tf.keras.layers.Dense(int(num_neurons/4), activation=activation)(x)
        x = tf.keras.layers.Dense(1, activation='linear')(x)
        v = tf.nn.softplus(x)

        # alpha
        x = latent
        x = tf.keras.layers.Dense(int(num_neurons/4), activation=activation)(x)
        x = tf.keras.layers.Dense(1, activation='linear')(x)
        alpha = tf.nn.softplus(x) + 1.

        # beta
        x = latent
        x = tf.keras.layers.Dense(int(num_neurons/4), activation=activation)(x)
        x = tf.keras.layers.Dense(1, activation='linear')(x)
        beta = tf.nn.softplus(x)

        model = tf.keras.Model(inputs=inputs, outputs=tf.concat([mu, v, alpha, beta], axis=-1))
        return model

    models = [create_model() for _ in range(num_ensembles)]

    return models, options
