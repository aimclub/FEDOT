FAQ
===

Frequently asked questions and answers

What is Fedot?
--------------

Fedot is the AutoML-like framework for the automated generation of the
data-driven composite models. It can solve classification, regression,
clustering, and forecasting problems.

Why should I use Fedot instead of existing state-of-the-art solutions (H2O/TPOT/etc)?
-------------------------------------------------------------------------------------

In practice, the existing AutoML solutions are really effective for the
limited set of problems only. During the model learning, modern AutoML
mostly focused on relatively simple tasks of hyperparameters
optimization, input data preprocessing, selecting a single model or a
set of models (this approach is also referred to as the Combined
Algorithm Selection and Hyperparameters optimization - CASH) since the
overall learning and meta-learning process is extremely expensive. In
the Fedot we have used the composite models concept. We claim,
that it allows us to solve many actual real-world problems in a more
efficient way. Also, we are aimed to outperform the existing solutions
even for well-known benchmarks (e.g. PMLB datasets).

Can I install Fedot using pip/conda?
------------------------------------

`Yes <https://pypi.org/project/fedot>`__

Why *feature_name* is not supported?
------------------------------------

We provide a constant extension of Fedot’s feature set. However, any
Pull Requests and issues from external contributors that introduce or
suggests the new features will be appreciated. You can create your `pull
request`_ or `issue`_ in the main repository of Fedot.

Can I use Fedot in my project/research/etc?
-------------------------------------------

Yes, you can. The Fedot is published under the BSD-3 license. Also, we
will be happy to help the users to adopt Fedot to their needs.

Why it is named Fedot?
----------------------

We decided to use this archaic Russian first name to add a bit of
fantasy spirit into the development process.

.. _pull request: https://github.com/nccr-itmo/FEDOT/pulls
.. _issue: https://github.com/nccr-itmo/FEDOT/issues