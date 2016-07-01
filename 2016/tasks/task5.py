import numpy as np


class Autoencoder(object):
    def __init__(self, list_layer_dim, eta=0.01):
        self.W, self.b = self._glorot_uniform(list_layer_dim)
        self._eta = eta

    def fit(self, xs, xs_validation, nb_epoch=10):
        history = {"train_loss": [], "validation_loss": []}

        for _ in range(nb_epoch):
            np.random.shuffle(xs)

            total_loss = 0.
            for x in xs:
                total_loss += self._train_on_batch(x)

            history["train_loss"].append(total_loss / len(xs))
            history["validation_loss"].append(self.evaluate(xs_validation))

        history["train_loss"] = np.array(history["train_loss"])
        history["validation_loss"] = np.array(history["validation_loss"])
        return history

    def evaluate(self, xs):
        total_loss = 0.
        for x in xs:
            z = x
            for W_n, b_n in zip(self.W, self.b):
                z = W_n.dot(z) + b_n

            total_loss += sum(np.square(x-z)) / 2
        return total_loss / len(xs)

    def _train_on_batch(self, x):
        Z, delta_N, loss = self._forward(x)
        deltas = self._backward(delta_N)
        self._update(Z, deltas)
        return loss

    def _forward(self, x):
        Z = [x]  # outputs of layer
        for i, (W_n, b_n) in enumerate(zip(self.W, self.b)):
            z = W_n.dot(Z[i]) + b_n
            Z.append(z)

        delta = Z[-1] - x
        loss = sum(delta**2) / 2
        return Z, delta, loss

    def _backward(self, output_layer_delta):
        deltas = [output_layer_delta]
        for i in range(len(self.W)-1, 0, -1):
            delta_i = self.W[i].transpose().dot(deltas[0])
            deltas.insert(0, delta_i)

        return deltas

    def _update(self, layer_inputs, deltas):
        for i in range(len(deltas) - 1, -1, -1):
            self.W[i] -= self._eta * np.outer(deltas[i], layer_inputs[i])
            self.b[i] -= self._eta * deltas[i]

    def _glorot_uniform(self, list_layers):
        W = []
        b = []
        for i in range(len(list_layers)-1):
            nb_pre_unit = list_layers[i]
            nb_next_unit = list_layers[i+1]
            upper = 1. / np.sqrt(nb_pre_unit)
            W.append(np.random.uniform(
                low=-upper, high=upper, size=(nb_next_unit, nb_pre_unit)))
            b.append(np.zeros(nb_next_unit))

        return W, b


class DenoisingAutoencoder(Autoencoder):
    """
    Denoising Auto Encoder with gaussian noise
    """

    def __init__(self, list_layer_dim, eta=0.01, cov_coefficient=0.):
        self.cov_coefficient = cov_coefficient
        super(DenoisingAutoencoder, self).__init__(
            list_layer_dim=list_layer_dim, eta=eta)

    def _train_on_batch(self, x):
        # add gaussian noise
        x += np.random.multivariate_normal(
            mean=np.zeros(len(x)), cov=self.cov_coefficient*np.eye(len(x)))
        return super(DenoisingAutoencoder, self)._train_on_batch(x=x)


if __name__ == '__main__':
    from sklearn.cross_validation import KFold
    np.random.seed(13)

    def format_for_csv(*args):
        """
        transform args to one line of csv
        """
        return ",".join(map(str, args)) + "\n"

    #  load data
    data_path = "../dataset.dat"
    f = open(data_path, "r")
    N, D = list(map(int, f.readline().split()))  # read header
    xs = np.array([list(map(float, l.split())) for l in f.readlines()])
    f.close()

    # tuning params
    list_layers = [[D,  5, D],
                   [D,  7, D],
                   [D, 13, D],
                   [D,  5, 5, D],
                   [D,  7, 7, D],
                   [D,  9, 9, D],
                   [D,  8, 5, 8, D],
                   [D,  7, 7, 7, D]]

    cov_coefficients = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])

    nb_fold = 5
    nb_epoch = 100

    out_file = open("task5_result.csv", "w")
    out_file.write("epoch,layer,cov,train_loss,validation_loss\n")

    # training Auto Encoder
    for layers in list_layers:
        string_layer = "-".join(map(str, layers))

        for train_indices, validation_indices in KFold(
                n=N, n_folds=nb_fold, shuffle=True, random_state=13):

            xs_train = xs[train_indices]
            xs_validation = xs[validation_indices]

            model = Autoencoder(list_layer_dim=layers)
            history = model.fit(
                xs=xs_train, xs_validation=xs_validation, nb_epoch=nb_epoch)

            # output learning results
            train_losses = history["train_loss"]
            validation_losses = history["validation_loss"]
            for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, validation_losses)):
                out_file.write(format_for_csv(epoch, string_layer, 0, t_loss, v_loss))

    # training Denoising Auto Encoder
    for layers in list_layers:
        string_layer = "-".join(map(str, layers))

        for cov_coefficient in cov_coefficients:
            for train_indices, validation_indices in KFold(
                    n=N, n_folds=nb_fold, shuffle=True, random_state=13):

                xs_train = xs[train_indices]
                xs_validation = xs[validation_indices]

                model = DenoisingAutoencoder(
                    list_layer_dim=layers, cov_coefficient=cov_coefficient)
                history = model.fit(
                    xs=xs_train, xs_validation=xs_validation, nb_epoch=nb_epoch)

                # output learning result
                train_losses, validation_losses = history["train_loss"], history["validation_loss"]
                for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, validation_losses)):
                    out_file.write(format_for_csv(epoch, string_layer, cov_coefficient, t_loss, v_loss))
