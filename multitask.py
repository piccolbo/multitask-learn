
#' If you missed my blog post on multitask learning and want to follow this
#' notebook, I highly recommend you start there. Here we'll perform a few
#' experiments to discuss the applicability of the new loss function and to show
#' how to implement it in Keras, a Python library that implements neural nets.
#' We will start with a set of tasks that ideally fulfill the assumptions under
#' which the loss function was derived and show the increased performance, at
#' least for this example. Then we will follow with a different examples that
#' doesn't fit so well the assumptions, and show that it doesn't work as well.

#' ##Learning the $\sin$ function

#' We'll keep it simple by choosing to learn the `sin` with different amount of
#' gaussian noise superimposed. We'll start with some standard imports:

import pandas as pd
import math
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ggplot as gg

#' Then we are going to generate some data randomly in the `[0,1]` interval:

N = 2000
xx = np.random.uniform(0, 1, N)
y = [math.sin(x * 2 * math.pi) for x in xx]
n_tasks = 5

#' We then pick 5 numbers to use as standard deviations and then draw from
#' normal distributions with 0 means and these standard deviations to obtain 5
#' ever more noisy versions of the original task. We finally add tow columns
#' with the values of the independent variable and the original, noiseless
#' dependent variable values, for ease of plotting:

sds = [.1 * 2**i for i in range(0, n_tasks)]
data = pd.DataFrame(
    {i: y + np.random.normal(0, sds[i], N)
     for i in range(len(sds))})
data.columns = ["y" + str(c) for c in data.columns]
data["x"] = xx
data["y"] = y

#' Let's take a look: we have 7 columns, 5 with the dependent variables, which
#' we will try to learn simultaneously, one for the dependent variable and one
#' for the noiseless values on which the five tasks are based:

data.head()

#' Admittedly, an ideal situation as all the assumptions on which the multitask
#' loss function is based are fulfilled. Also, the 5 learning tasks are very
#' closely related, as the "learnable" part of each task is the same, even if
#' the
#' data looks different. Let's split the data in train, test and stop data set.
#' I
#' need a stop dataset because I plan to use *early stopping* to protect
#' against
#' overfitting, but in many papers the test set is used for such purpose. I
#' believe a true test set and a learning alogrithm need to have an "air gap"
#' for the test set to fulfil its task, as many Netflix competitions have amply
#' demonstrated. But I digress.


def split_data(data):
    """Split a dataset in three chunks in 1:9:10 proportions.

    Parameters
    ----------
    data : type
        DataFrame to split.

    Returns
    -------
    type
        A tuple with three data frames.

    """
    N = data.shape[0]
    return data[0:N // 20], data[N // 20:N // 2], data[N // 2:N]


data_train, data_stop, data_test = split_data(data)


#' You may have notived that stop and test set are much bigger then the training
#' set. I just wanted to keep the task difficult but have a less noisy way to avoid
#' overfitting and to evaluate the effect of the change in loss function, which is
#' the only goal of this exercise. In practice, one would not use a dataset this
#' way.

#' Let's define a simple plotting function to see what's in the data set, and take
#' a peek:


def facet_plot(data):
    """Generate a line plot with one facet per column from a data frame.

    Parameters
    ----------
    data : type
        A data frame with one column named x as the independent variable.

    Returns
    -------
    type
        None, but displays graphics.

    """
    print(gg.ggplot(
        data=data.melt(id_vars=["x"]), aesthetics=gg.aes(x="x", y="value")) +
          gg.geom_point(size=2) + gg.facet_wrap("variable"))


facet_plot(data_test)

#' As you can see, progressively more noisy versions of the same task. If we look
#' at the training set in the same way, the task look a lot harder. `y3` and `y4`
#' may look hopeless, even:

facet_plot(data_train)

#' Let's import the neural net-related modules. The surprisingly named
#' `keras.backend` is a module containing lower level operations, compared to the
#' neural net abstraction, like sums and products, while providing *backend
#' abstraction*. A backend for Keras is one of several neural net libraries that
#' implement the machine learning algorithms all the way to the metal, including,
#' optionally, the GPU. Keras is just a thin layer on top of them, but a very well
#' designed one. For this work, if you are interested, the backend was Theano.


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import keras as K
import keras.backend as kb

#' Let's now define out neural net model. I picked a multi-layer perceptron
#' with 10 layers so as to allow enough steps for the task to be solved in an
#' integrated fashion. One big novelty about deep learning is that we have the
#' possibility of sharing internal representations between tasks, whereas with
#' shallow models, neural or not, this is not possible. I did not tinker much
#' with sizing since I didn't want to bias the results of the experiments. One
#' thing I did tinker with a little, reluctantly, were the `patience` parameter
#' in the *early stopping* rule and the `epsilon` parameter in the *Adam*
#' optimizer. This was to counter a pernicious tendency of the optimization to
#' reverse course, that is increase the loss, some time drastically, towards
#' the end of the fitting process. Apparently the Adam optimizer is prone, when
#' gradients are very small, to running into numerical stability issues, and
#' increasing the epsilon parameter helps with that, while slowing the learning
#' process. Increasing the `patience` parameter has the effect of continuing
#' the optimization even when *plateaus* of very low gradient are reached.
#' Decreasing it results in the fitting process to stop before it has reached a
#' local minumum, because of the randomness intrinsic to Stochastic Gradient
#' Descent, on which Adam is based. It is unfortunate the all these optimizers
#' have so many knobs that may need to be accessed to achieve good performance.
#' Not only they are laborious to operate, but they also cast a shadow of
#' "secret sauce" on these methods, meaning that they don't work reliably unless
#' considerable know-how is provided by the practicioner.


def NN_experiment(data_train, data_stop, loss):
    U = 100
    L = 10
    NN = K.models.Sequential([
        K.layers.Dense(
            input_shape=(1, ), units=U, activation=K.activations.relu)
    ] + [
        K.layers.Dense(units=U, activation=K.activations.relu)
        for _ in range(L - 1)
    ] + [K.layers.Dense(n_tasks, activation="linear")])
    NN.compile(optimizer=K.optimizers.Adam(epsilon=1E-3), loss=loss)
    hist = NN.fit(
        x=data_train["x"].values,
        y=data_train.iloc[:, range(0, n_tasks)].values,
        epochs=200,
        callbacks=[K.callbacks.EarlyStopping(patience=5)],
        validation_data=(data_stop["x"].values,
                         data_stop.iloc[:, range(0, n_tasks)].values),
        verbose=0)
    return (NN, hist)


NN_homo, hist_homo = NN_experiment(data_train, data_stop,
                                   K.losses.mean_squared_error)


def loss_plot(hist):
    hist.history["epoch"] = hist.epoch
    data = pd.DataFrame(hist.history)
    print(gg.ggplot(
        data=data.melt(id_vars=["epoch"], var_name="metric"),
        aesthetics=gg.aes(x="epoch", y="value", color="metric")) +
          gg.scale_y_log() + gg.geom_line())


#'  Some text

loss_plot(hist_homo)

#' Some text


def make_pred(NN, data, target):
    pred = pd.DataFrame(NN.predict(x=data["x"]), index=data.index)
    pred.columns = ["y" + str(c) for c in pred.columns]
    pred["x"] = data["x"]
    return pd.concat([pred, target], axis=1, join="inner")


#' Some text


def multi_line_plot(data):
    print(gg.ggplot(
        data=data.melt(id_vars=["x"]),
        aesthetics=gg.aes(x="x", y="value", color="variable")) +
          gg.geom_line())


#' Some text

multi_line_plot(make_pred(NN_homo, data_test, pd.DataFrame({"y": y})))

#' Some text


def mean_squared_error_hetero(y_true, y_pred):
    return kb.exp(
        kb.mean(kb.log(kb.mean(kb.square(y_pred - y_true), axis=0)), axis=-1))


#' Some text

NN_hetero, hist_hetero = NN_experiment(data_train, data_stop,
                                       mean_squared_error_hetero)

#' Some text

loss_plot(hist_hetero)

#' Some text

multi_line_plot(make_pred(NN_hetero, data_test, pd.DataFrame({"y": y})))

#' Some text

comp_perf = pd.DataFrame([
    make_pred(NN, data_test, pd.DataFrame({
        "y": y
    })).subtract(data_test["y"], axis=0).iloc[:, range(5)].pow(2).mean()
    for NN in (NN_homo, NN_hetero)
])
comp_perf

#' Some text

comp_perf.iloc[0] / comp_perf.iloc[1]

#' Some text

import pickle
import os
os.chdir('../multitask-learn')
with open("pstats.cpickle", "rb") as f:
    pstats = pickle.load(f, encoding='latin1')
pstats.head()

#' Some text

#%matplotlib inline
pstats.plot.box(logy=True)

#' Some text

data_harmo = pd.DataFrame({
    "y" + str(i):
    [math.sin((x * 2 + float(i) / n_tasks) * math.pi) for x in xx]
    for i in range(n_tasks)
})
data_harmo["x"] = xx
data_harmo.head()

#' Some text

data_harmo_train, data_harmo_stop, data_harmo_test = split_data(data_harmo)

#' Some text

facet_plot(data_harmo_test)

#' Some text

facet_plot(data_harmo_train)

#' Some text

NN_harmo, hist_harmo = NN_experiment(data_harmo_train, data_harmo_stop,
                                     K.losses.mean_squared_error)

#' Some text

loss_plot(hist_harmo)

#' Some text

facet_plot(make_pred(NN_harmo, data_harmo_test, None))

#' Some text

NN_hh, hist_hh = NN_experiment(data_harmo_train, data_harmo_stop,
                               mean_squared_error_hetero)

#' Some text

loss_plot(hist_hh)

#' Some text

facet_plot(make_pred(NN_hh, data_harmo_test, data_harmo_test))
