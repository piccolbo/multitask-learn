"""Supporting library for multitask loss function post."""

from joblib import Memory
import keras as K
import keras.backend as kb
import keras.losses as kl
import math
import numpy as np
import pandas as pd
import sys
import tabulate as tab

sys.setrecursionlimit(20000)  # for joblib.memory to work
memory = Memory(cachedir=".joblib", verbose=0)

n_tasks = 5
N = 2000


def make_data_hetero():
    """Generate data for the heteroscedastic multitask learning experiment.

    Returns
    -------
    DataFrame
        One column per learning task, increasing noise

    """
    xx = np.random.uniform(0, 1, N)
    y = [math.sin(x * 2 * math.pi) for x in xx]

    sds = [.1 * 2**i for i in range(0, n_tasks)]
    data = pd.DataFrame(
        {i: y + np.random.normal(0, sds[i], N)
         for i in range(len(sds))})
    data.columns = ["y" + str(c) for c in data.columns]
    data["x"] = xx
    data["y"] = y
    return data


def make_data_phase():
    """Generate data for the multiphase example.

    Returns
    -------
    DataFrame
        Data f.

    """
    xx = np.random.uniform(0, 1, N)
    data = pd.DataFrame({
        "y" + str(i):
        [math.sin((x * 2 + float(i) / n_tasks) * math.pi) for x in xx]
        for i in range(n_tasks)
    })
    data["x"] = xx
    return data


def mean_squared_error_hetero(y_true, y_pred):
    """Loss function for multitask learning.

    Parameters
    ----------
    y_true : tensor
        The target value/
    y_pred : tensor
        The predicted value.

    Returns
    -------
    tensor
        Distance.

    """
    return kb.exp(
        kb.mean(kb.log(kb.mean(kb.square(y_pred - y_true), axis=0)), axis=-1))


def NN_experiment(data_train, data_stop, loss):
    """Fit a model.

    Parameters
    ----------
    data_train : DataFrame
        Data to train on.
    data_stop : DataFrame
        Data to use for early stopping.
    loss : Function
        Loss function to use.

    Returns
    -------
    Tuple of Keras Model and hist object
        Tuple with fitted model and history of fitting process.

    """
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
    # importing on demand to avoid issues on OS X with frameworks
    import ggplot as gg
    print(
        gg.ggplot(
            data=data.melt(id_vars=["x"]), aesthetics=gg.aes(x="x", y="value"))
        + gg.geom_point(size=2) + gg.facet_wrap("variable"))


def loss_plot(hist):
    """Plot train and validation loss during training.

    Parameters
    ----------
    hist : Keras hist object.
        History of the fitting process.

    Returns
    -------
    NoneType
        None.

    """
    # importing on demand to avoid issues on OS X with frameworks
    import ggplot as gg
    hist.history["epoch"] = hist.epoch
    data = pd.DataFrame(hist.history)
    print(
        gg.ggplot(
            data=data.melt(id_vars=["epoch"], var_name="metric"),
            aesthetics=gg.aes(x="epoch", y="value", color="metric")) +
        gg.scale_y_log() + gg.geom_line())


def multi_line_plot(data, target):
    """Make a multi-line plot of data and target.

    Parameters
    ----------
    data : DataFrame
        Data to plot.
    target: Series or DataFrame
        Additional data to plot.

    Returns
    -------
    NoneType
        None.

    """
    # importing on demand to avoid issues on OS X with frameworks
    import ggplot as gg
    data = pd.concat([data, target], axis=1, join="inner")
    print(
        gg.ggplot(
            data=data.melt(id_vars=["x"]),
            aesthetics=gg.aes(x="x", y="value", color="variable")) +
        gg.geom_line())


def make_pred(NN, x):
    """Make predictions.

    Parameters
    ----------
    NN : Keras model
        Model to use in prediction.
    x : Series
        Values for independent variable.

    Returns
    -------
    DataFrame
        Predictions.

    """
    pred = pd.DataFrame(NN.predict(x=x), index=x.index)
    pred.columns = ["y" + str(c) for c in pred.columns]
    return pd.concat([pred, x], axis=1, join="inner")


def compare_performance(NN1, NN2, data_test, target):
    """Compare performance of two models.

    Parameters
    ----------
    NN1 : keras model
        First  model to compare.
    NN2 : keras model
        Second model to compare.
    data_test : type
        Data to perform comparison  on.
    y : Pandas series
        Ground truth.

    Returns
    -------
    Series
        MSE for each prediction task.

    """
    return pd.DataFrame([
        make_pred(NN, data_test["x"]).subtract(target, axis=0).pow(2).mean()
        for NN in (NN1, NN2)
    ]).drop(
        "x", axis=1)


def one_replication(make_data):
    """Perform one replication of the loss function comparison.

    Parameters
    ----------
    make_data : function
        Generator for data to perform the experiment on.

    Returns
    -------
    Series
        MSE ratio, task by task.

    """
    data = make_data()
    data_train, data_stop, data_test = split_data(data)
    NN_mse, _ = NN_experiment(data_train, data_stop, kl.mean_squared_error)
    NN_mseh, _ = NN_experiment(data_train, data_stop,
                               mean_squared_error_hetero)
    cperf = compare_performance(NN_mse, NN_mseh, data_test, data_test["y"])
    return cperf.iloc[0] / cperf.iloc[1]


def many_replications(make_data, n=100):
    """Perform many replications of the loss comparison.

    Parameters
    ----------
    make_data : Function.
        Data generator for experiment.
    n : int
        Number of replications.

    Returns
    -------
    DataFrame
        MSE ratio, one col per task, one row per replication.

    """
    return pd.DataFrame([one_replication(make_data_hetero) for _ in range(n)])


many_replications_ = memory.cache(many_replications)

pd.set_option('precision', 2)


def print_table(df):
    """Short summary.

    Parameters
    ----------
    df : type
        Description of parameter `df`.

    Returns
    -------
    type
        Description of returned object.

    """
    print(
        tab.tabulate(
            df,
            headers="keys",
            showindex=False,
            floatfmt=".2f",
            tablefmt="pipe"))
