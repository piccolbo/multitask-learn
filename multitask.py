#' ---
#' layout: "post"
#' title: "A Simple Loss Function for Multi-Task learning with Keras implementation, part&nbsp;2"
#' date: "2018-03-23 10:45"
#' ---



#' In this post, we show how to implement a custom loss function for multitask
#' learning in Keras and perform a couple of simple experiments with itself.
#' <!-- more -->

#' In a [previous
#' post](/2018/03/a-simple-loss-function-for-multi-task-learning-with-keras-implementation.html),
#' I filled in some details of [recent work](https://arxiv.org/abs/1705.07115)
#' on on multitask learning. Here we'll perform a few experiments to discuss the
#' applicability of the new loss function described therein and show an
#' implementation in Keras, a Python library for neural networks.  We will start
#' with a set of tasks that ideally fulfill the assumptions under which the loss
#' function was derived and show the increased performance. Then we will follow
#' with a different examples that doesn't fit so well the assumptions, and show
#' that it doesn't work as well.  Only the ML-related code is shown here, but
#' the rest is [available](https://github.com/piccolbo/multitask)

#+ echo=False, results="hidden"
import numpy as np
import pandas as pd
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ggplot  # only to prevent warnings elsewhere
    import keras.losses as kl
    # workaround for pwave path issue
    sys.path.insert(0, ".")
    import multitask_lib as mtl

# import importlib as il
# il.reload(mtl)

np.random.seed(seed=1)

#' ## Learning the $$\sin$$ function with noise

#' We'll keep it simple by choosing to learn the $$\sin$$ function in the $$[0,
#' 2\pi]$$ interval with different amounts of gaussian noise added.  The first
#' task is generating some data to feed into this experiment.  We sample the
#' independent variable to randomly in the $$[0,1]$$ interval,  then pick 5
#' numbers to use as standard deviations and then draw from normal distributions
#' with 0 means and these standard deviations to obtain 5 progressively more
#' noisy versions of the original task. We finally add two columns with the
#' values of the independent variable and the original, noiseless dependent
#' variable values, for ease of plotting (`np` is short for `numpy` and `pd` for
#' `pandas`):

#+ source="mtl.make_data_hetero"

data_hetero = mtl.make_data_hetero()

#' Let's take a look: we have 7 columns, 5 with the dependent variables, which
#' we will try to learn simultaneously, one for the dependent variable and one
#' for the noiseless values on which the five tasks are based:
#+ results="raw"
mtl.print_table(data_hetero.head())


#' While discussing multitask learning is not the goal here, this is a
#' favorable setting for it, with 5 closely related tasks in that the target
#' values are the same for all the tasks. But the 5 tasks are also different
#' since they are progressively more noisy, a problem the multitask loss here
#' presented is designed to tackle.  Let's split the data in train, test and
#' stop data set.  I need a stop dataset because I plan to use *early stopping*
#' to protect against overfitting, but in many papers the test set is used for
#' such purpose. I believe a true test set and a learning algorithm need to have
#' an "air gap" for the test set to fulfil its task, as many Kaggle competitions
#' have amply
#' [demonstrated](http://blog.kaggle.com/2012/07/06/the-dangers-of-overfitting-psychopathy-post-mortem/).

#+ source="mtl.split_data"
data_hetero_train, data_hetero_stop, data_hetero_test = mtl.split_data(
    data_hetero)

#' You may have noticed that stop and test sets are much bigger then the
#' training set. I  wanted to keep the task difficult but have  reliable ways to
#' avoid overfitting and to evaluate the effect of the change in loss function,
#' which is the only goal of this exercise. In practice, such a split is not
#' commonly used.

#' First let's take a peek at the training set:

#+ results="hidden", echo = False
mtl.facet_plot(data_hetero_train.drop(["y"], axis=1))

#' As you can see, progressively more noisy versions of the same task.

#' Let's now define a neural net model. I picked a multi-layer perceptron with
#' 10 layers so as to allow enough steps for the tasks to be solved in an
#' integrated fashion. One big novelty about [deep
#' learning](http://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf)
#' is that we have the possibility of [sharing internal
#' representations](http://papers.nips.cc/paper/959-learning-many-related-tasks-at-the-same-time-with-backpropagation.pdf)
#' between tasks, whereas with shallow models, neural or not, this is not
#' possible. I did not tinker much with sizing, since I didn't want to bias the
#' results of the experiments.  One thing I did tinker with a little,
#' reluctantly, were the `patience` parameter in the *early stopping* rule and
#' the `epsilon` parameter in the *Adam* optimizer. This was to counter a
#' pernicious tendency of the optimization to reverse course, that is increase
#' the loss, some time drastically, towards the end of the fitting process.
#' Apparently the Adam optimizer is
#' [prone](https://github.com/pytorch/pytorch/issues/1767), when gradients are
#' very small, to running into numerical stability issues, and increasing the
#' epsilon parameter helps with that, while slowing the learning process.
#' Increasing the `patience` parameter has the effect of continuing the
#' optimization even when *plateaus* of very low gradient are reached.
#' Decreasing it results in the fitting process to stop before it has reached a
#' local minumum, because of the randomness intrinsic to Stochastic Gradient
#' Descent, on which Adam is based. It is unfortunate that many if not all of
#' these optimizers have so many knobs that may need to be accessed to achieve
#' good performance.  Not only are they laborious to operate, but they can also
#' get us lost in Andrew Gelman's so-called [Garden of Forking Paths](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.709.5937&rep=rep1&type=pdf), whereby
#' we can run into overfitting without being aware of it.  The importance of
#' this is starting to be [recognized](https://twitter.com/goodfellow_ian/status/978342148402593792) by the most alert members of the ML
#' community, but is a well established theme in statistics (multiple testing
#' correction, false discovery rate etc.)

#+ source="mtl.NN_experiment"
NN_hetero_mse, hist_hetero_mse = mtl.NN_experiment(
    data_hetero_train, data_hetero_stop, kl.mean_squared_error)

#' We are storing not only the fitted model, but also a record of the training
#' process, which enables a standard visualization of how training set
#' and stop set losses (named `loss` and `val_loss` resp. in the graph) decrease
#' over the epochs of training. Also please note how this function is
#' parametrized on the loss function, which will allow us to perform the central
#' comparison in this post.
#+ results="hidden", echo = False
mtl.loss_plot(hist_hetero_mse)

#' And here is a look at the actual predictions. You be the judge if they are
#' any good; later we will try to compare this and the novel approach
#' in a more rigorous way.
#+ results="hidden", echo = False
mtl.multi_line_plot(
    mtl.make_pred(NN_hetero_mse, data_hetero_test["x"]), data_hetero_test["y"])

#' Now we are going to repeat this is experiment substituting the standard MSE
#' loss with the one derived in the previous post on this subject. As you can
#' see the implementation in Keras is simple but one has to substitute standard vector
#' operations with  Keras low level ones, and make sure to use the correct
#' *axis* when operating on tensors.
#+ source="mtl.mean_squared_error_hetero"
NN_hetero_mseh, hist_hetero_mseh = mtl.NN_experiment(
    data_hetero_train, data_hetero_stop, mtl.mean_squared_error_hetero)

#' The same loss plot as before for completeness:

#+ results="hidden", echo = False
mtl.loss_plot(hist_hetero_mseh)

#' And the line plot:
#+ results="hidden", echo = False
mtl.multi_line_plot(
    mtl.make_pred(NN_hetero_mseh, data_hetero_test["x"]),
    data_hetero_test["y"])

#' Is it better? Let's look into it:

#+ source="mtl.compare_performance"
cperf = mtl.compare_performance(NN_hetero_mse, NN_hetero_mseh,
                                data_hetero_test, data_hetero_test["y"])

#' And, comparing task by task, it looks like it mostly is:

cperf.iloc[0] / cperf.iloc[1]

#' The most alert readers for sure are aware of the [reproducibility
#' crisis](https://en.wikipedia.org/wiki/Replication_crisis) in the sciences,
#' whereby many published results do not stand the test of time.  This is a
#' [complex subject](http://scienceincrisis.info/), but one important factor is
#' that sometimes surprising and interesting and therefore very publishable
#' results occur by chance.  Sometimes luck is enhanced, deliberately or not, by
#' the researcher, operating on various analysis "knobs" or discarding "bad
#' data" (also known with the suave term of "data cleaning") or modifying the
#' metric of interest or the research question until something "interesting" but
#' random shows up.  The reviewing and publication process creates the wrong
#' incentives by insisting more on novelty than methodological rigor.  In AI
#' research [this problem](http://science.sciencemag.org/content/359/6377/725)
#' is compounded by frantic competition, automation of hyperparameter search and
#' by the dominance of a few very prominent benchmark datasets that have been
#' studied in depth, therefore there isn't any data that can be used as a
#' separate test set anymore. In this post, working on a small synthetic
#' dataset, we have the opportunity to repeat the experiment many times and see
#' what happens:
#+ source = "mtl.one_replication", results = "hidden"
_
#+ source = "mtl.many_replications", results="hidden"
pstats_hetero = mtl.many_replications_(mtl.make_data_hetero)
pstats_hetero.plot.box(logy=True)

#' Now we have a better understanding of how the new loss function helps in
#' majority of cases, but can also hurt sometimes.

#' ## Learning the $$\sin$$ function with multiple phases

#' We change tack now by tackling with the same techniques a problem where, like
#' in many AI problems, the challenge is not randomness in the data but just the
#' complexity of the tasks. Specifically, we are going to try to fit a neural
#' model to 5 shifted versions of the the $$\sin$$ function, with no added noise.
#+ source ="mtl.make_data_phase"
data_phase = mtl.make_data_phase()

#' We split the data exactly as before:

data_phase_train, data_phase_stop, data_phase_test = mtl.split_data(data_phase)

#' Let's take a look at the training data. Here the 5 tasks are equally
#' difficult and, since the data is noise-free, it's more a function
#' approximation problem than a statistical problem.
#+ results="hidden", echo = False
mtl.facet_plot(data_phase_train)

#' We can reuse the same code as in the first part of this post, starting with
#' the standard MSE loss:

NN_phase_mse, hist_phase_mse = mtl.NN_experiment(
    data_phase_train, data_phase_stop, kl.mean_squared_error)

#' We examine as usual the loss dynamic to diagnose any problems in the fitting
#' process:
#+ results="hidden", echo = False
mtl.loss_plot(hist_phase_mse)

#' Next, we take a look at the results:
#+ results="hidden", echo = False
mtl.facet_plot(mtl.make_pred(NN_phase_mse, data_phase_test["x"]))

#' Now we switch to the multitask loss:

NN_phase_mseh, hist_phase_mseh = mtl.NN_experiment(
    data_phase_train, data_phase_stop, mtl.mean_squared_error_hetero)

#' Take a look at the loss plot (please note some instability at the end of the
#' process, as noted before a known problem with the Adam optimizer)
#+ results="hidden", echo = False
mtl.loss_plot(hist_phase_mseh)

#' And a look at the predictions:
#+ results="hidden", echo = False
mtl.facet_plot(mtl.make_pred(NN_phase_mseh, data_phase_test["x"]))

#' They don't look great and a quantitative assessment confirms that impression:

cperf = mtl.compare_performance(NN_phase_mse, NN_phase_mseh, data_phase_test,
                                data_phase_test)
cperf.iloc[0] / cperf.iloc[1]

#' Was it a fluke? Replication to the rescue!
#+ results = "hidden"
pstats_phase = mtl.many_replications_(mtl.make_data_phase)
pstats_phase.plot.box(logy=True)

#' It does indeed look like the first run was not particularly lucky. The
#' boxplot shows instead that the multitask loss outperforms the MSE loss even
#' when there is no difference between the tasks in this particular example
#' &mdash; which, we need to remark, is qualitatively very different from the
#' image analysis application for which this idea was developed.

#' ## Conclusions
