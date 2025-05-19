Abstract
========

.. topic:: What is FEDOT?

    *FEDOT is the AutoML-like framework for the automated generation of the
    data-driven composite models. It can solve classification, regression,
    clustering, and forecasting problems.*

.. topic:: Why FEDOT is framework?

    *While the exact difference between 'library' and 'framework' is a bit ambiguous and
    context-dependent in many cases, we still consider FEDOT as a framework.*

    *The reason is that is can be used not only to solve pre-defined AutoML task,
    but also can be used to build new derivative solutions.
    As an examples:* `FEDOT.NAS`_, `FEDOT.Industrial`_.

.. topic:: Why should I use FEDOT instead of existing state-of-the-art solutions (LightAutoML/AutoGluon/H2O/etc)?

    *In practice, the existing AutoML solutions are really effective for the
    limited set of problems only. During the model learning, modern AutoML
    mostly focused on relatively simple tasks of hyperparameters
    optimization, input data preprocessing, selecting a single model or a
    set of models (this approach is also referred to as the Combined
    Algorithm Selection and Hyperparameters optimization - CASH) since the
    overall learning and meta-learning process is extremely expensive. In
    the FEDOT we have used the composite models concept. We claim,
    that it allows us to solve many actual real-world problems in a more
    efficient way. Also, we are aimed to outperform the existing solutions
    even for well-known benchmarks (e.g. PMLB datasets).*

.. topic:: Can I install FEDOT using pip/conda?

    *Yes, follow the* `link`_.

.. topic:: Can I use FEDOT in my project/research/etc?

    *Yes, you can. The Fedot is published under the BSD-3 license. Also, we
    will be happy to help the users to adopt Fedot to their needs.*

.. topic:: Why it is named FEDOT?

    *We decided to use this archaic Russian first name to add a bit of
    fantasy spirit into the development process.*


.. List of links:

.. _link: https://pypi.org/project/fedot
.. `link` replace:: *link*

.. _FEDOT.NAS: https://github.com/ITMO-NSS-team/nas-fedot
.. `FEDOT.NAS` replace:: *FEDOT.NAS*

.. _FEDOT.Industrial: https://github.com/aimclub/Fedot.Industrial
.. `FEDOT.Industrial` replace:: *FEDOT.Industrial*
