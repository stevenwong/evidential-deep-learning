import numpy as np
import tensorflow as tf
import time
import datetime
import os
import sys
import h5py
from pathlib import Path

import evidential_deep_learning as edl
from .util import normalize, gallery

class NormalGamma:
    def __init__(self, models, opts, dataset="", learning_rate=1e-3, lam=0.0, epsilon=1e-2, maxi_rate=1e-4, tag=""):
        self.nll_loss_function = edl.losses.SMD_NLL2
        # self.reg_loss_function = edl.losses.NIG_Reg

        self.models = models
        self.num_ensembles = opts['num_ensembles']
        self.learning_rate = learning_rate
        self.maxi_rate = maxi_rate

        self.optimizers = [tf.optimizers.Adam(learning_rate) for _ in range(self.num_ensembles)]
        # self.lam = tf.Variable(lam)

        self.epsilon = epsilon

        self.min_rmse = self.running_rmse = float('inf')
        self.min_nll = self.running_nll = float('inf')
        self.min_vloss = self.running_vloss = float('inf')

        trainer = self.__class__.__name__
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join('save','{}_{}_{}_{}'.format(current_time, dataset, trainer, tag))
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        train_log_dir = os.path.join('logs', '{}_{}_{}_{}_train'.format(current_time, dataset, trainer, tag))
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs', '{}_{}_{}_{}_val'.format(current_time, dataset, trainer, tag))
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def loss_function(self, y, mu, s2b, alpha, reduce=True, return_comps=False):
        nll_loss = self.nll_loss_function(y, mu, s2b, alpha, reduce=reduce)
        # reg_loss = self.reg_loss_function(y, mu, v, alpha, beta, reduce=reduce)
        # loss = nll_loss + self.lam * (reg_loss - self.epsilon)
        loss = nll_loss

        return (loss, (nll_loss, 0.)) if return_comps else loss

    # @tf.function
    # def run_train_step(self, x, y):
    #     with tf.GradientTape() as tape:
    #         outputs = self.model(x, training=True)
    #         mu, v, alpha, beta = tf.split(outputs, 4, axis=-1)
    #         loss, (nll_loss, reg_loss) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)

    #     grads = tape.gradient(loss, self.model.trainable_variables) #compute gradient
    #     self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #     # self.lam = self.lam.assign_add(self.maxi_rate * (reg_loss - self.epsilon)) #update lambda

    #     return loss, nll_loss, reg_loss, mu, v, alpha, beta

    @tf.function
    def run_train_step(self, x, y):
        losses = []
        y_hats = []
        for (model, optimizer) in zip(self.models, self.optimizers): #Autograph unrolls this so make sure ensemble size is not too large

            with tf.GradientTape() as tape:
                outputs = model(x, training=True) #forward pass
                mu, s2b, alpha = tf.split(outputs, 3, axis=-1)
                loss, (nll_loss, reg_loss) = self.loss_function(y, mu, s2b, alpha, return_comps=True)
                y_hats.append(mu)
                losses.append(loss)

            grads = tape.gradient(loss, model.variables) #compute gradient
            optimizer.apply_gradients(zip(grads, model.variables))

        return tf.reduce_mean(losses), tf.reduce_mean(nll_loss), tf.reduce_mean(reg_loss), tf.reduce_mean(y_hats, 0)

    # @tf.function
    # def evaluate(self, x, y):
    #     outputs = self.model(x, training=False)
    #     mu, v, alpha, beta = tf.split(outputs, 4, axis=-1)

    #     rmse = edl.losses.RMSE(y, mu)
    #     loss, (nll, reg_loss) = self.loss_function(y, mu, v, alpha, beta, return_comps=True)

    #     return mu, v, alpha, beta, loss, rmse, nll, reg_loss

    @tf.function
    def evaluate(self, x, y):
        preds = tf.stack([model(x, training=False) for model in self.models], axis=0) #forward pass
        mus, s2bs, alphas = tf.split(preds, 3, axis=-1)
        mean_mu = tf.reduce_mean(mus, axis=0)
        # var = s2bs / (alphas - 1.)
        var = s2bs * alphas / (alphas - 1.)
        var = tf.reduce_mean(var + tf.square(mus), axis=0) - tf.square(mean_mu)
        # loss = self.loss_function(y, mean_mu, tf.sqrt(var))

        rmse = edl.losses.RMSE(y, mean_mu)
        nll = edl.losses.Gaussian_NLL(y, mean_mu, tf.sqrt(var))

        return mean_mu, var, nll, rmse, nll

    def normalize(self, x):
        return tf.divide(tf.subtract(x, tf.reduce_min(x)),
               tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))


    def save_train_summary(self, loss, x, y, y_hat):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, y_hat)), step=self.iter)
            tf.summary.scalar('loss', tf.reduce_mean(loss), step=self.iter)
            idx = np.random.choice(int(tf.shape(x)[0]), 9)
            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(y_hat,idx).numpy())], max_outputs=1, step=self.iter)

    def save_val_summary(self, loss, x, y, mu, var):
        with self.val_summary_writer.as_default():
            tf.summary.scalar('mse', tf.reduce_mean(edl.losses.MSE(y, mu)), step=self.iter)
            tf.summary.scalar('loss', tf.reduce_mean(loss), step=self.iter)
            idx = np.random.choice(int(tf.shape(x)[0]), 9)
            if tf.shape(x).shape==4:
                tf.summary.image("x", [gallery(tf.gather(x,idx).numpy())], max_outputs=1, step=self.iter)

            if tf.shape(y).shape==4:
                tf.summary.image("y", [gallery(tf.gather(y,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_hat", [gallery(tf.gather(mu,idx).numpy())], max_outputs=1, step=self.iter)
                tf.summary.image("y_var", [gallery(normalize(tf.gather(var,idx)).numpy())], max_outputs=1, step=self.iter)

    def get_batch(self, x, y, batch_size):
        idx = np.random.choice(x.shape[0], batch_size, replace=False)
        if isinstance(x, tf.Tensor):
            x_ = x[idx,...]
            y_ = y[idx,...]
        elif isinstance(x, np.ndarray) or isinstance(x, h5py.Dataset):
            idx = np.sort(idx)
            x_ = x[idx,...]
            y_ = y[idx,...]

            x_divisor = 255. if x_.dtype == np.uint8 else 1.0
            y_divisor = 255. if y_.dtype == np.uint8 else 1.0

            x_ = tf.convert_to_tensor(x_/x_divisor, tf.float32)
            y_ = tf.convert_to_tensor(y_/y_divisor, tf.float32)
        else:
            print("unknown dataset type {} {}".format(type(x), type(y)))
        return x_, y_

    def save(self, name):
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.save_dir, "{}_{}.h5".format(name, i)))

    def update_running(self, previous, current, alpha=0.0):
        if previous == float('inf'):
            new = current
        else:
            new = alpha*previous + (1-alpha)*current
        return new

    def train(self, x_train, y_train, x_test, y_test, y_scale, batch_size=128, iters=10000, verbose=True):
        tic = time.time()
        for self.iter in range(iters):
            x_input_batch, y_input_batch = self.get_batch(x_train, y_train, batch_size)
            loss, nll_loss, reg_loss, y_hat = self.run_train_step(x_input_batch, y_input_batch)

            if self.iter % 10 == 0:
                self.save_train_summary(loss, x_input_batch, y_input_batch, y_hat)

            if self.iter % 100 == 0:
                x_test_batch, y_test_batch = self.get_batch(x_test, y_test, min(100, x_test.shape[0]))
                mu, var, vloss, rmse, nll = self.evaluate(x_test_batch, y_test_batch)

                nll += np.log(y_scale[0,0])
                rmse *= y_scale[0,0]

                self.save_val_summary(vloss, x_test_batch, y_test_batch, mu, var)

                if rmse.numpy() < self.min_rmse:
                    self.min_rmse = rmse.numpy()
                    self.save(f"model_rmse_{self.iter}")

                if nll.numpy() < self.min_nll:
                    self.min_nll = nll.numpy()
                    self.save(f"model_nll_{self.iter}")

                if vloss.numpy() < self.min_vloss:
                    self.min_vloss = vloss.numpy()
                    self.save(f"model_vloss_{self.iter}")

                if verbose: print("[{}] \t RMSE: {:.4f} \t NLL: {:.4f} \t train_loss: {:.4f} \t t: {:.2f} sec".format(self.iter, self.min_rmse, self.min_nll, loss, time.time()-tic))
                tic = time.time()

        return self.models, self.min_rmse, self.min_nll
