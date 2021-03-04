Sensitivity analysis
--------------------

Sensitivity analysis algorithms are used to find uncertainties in the input parameters
of data or in meta-parameters that affect the output. But the model complexity
can also affect the result. To evaluate the potential contribution of certain component
of the composite model we introduce the composite model Structure Analysis.

Structural Analysis means structure modification of some sort like Node deletion,
replacement or anything else(see `NodeAnalyzeApproach`_). Within FEDOT, you can apply such an analysis
to the model by using `ChainStructureAnalysis`_ or `NodeAnalysis`_. The difference is:

- ``ChainStructureAnalysis`` allows you to apply several approaches to several (all or defined) Nodes;

- ``NodeAnalysis`` allows you to apply several approaches to one Node;

- additionally, you may use every approach directly.

ChainStructureAnalysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

     from fedot.core.chains.chain import Chain
     from fedot.core.data.data import InputData
     from fedot.sensitivity.chain_sensitivity import ChainStructureAnalyze
     from fedot.sensitivity.node_sensitivity import \
     NodeDeletionAnalyze, NodeReplaceModelAnalyze

     chain = Chain()
     train_data = InputData.from_csv('train_file.csv')
     test_data = InputData.from_csv('test_file.csv')
     approaches = [NodeDeletionAnalyze, NodeReplaceModelAnalyze]
     analysis_result = ChainStructureAnalyze(chain=chain,
                                             train_data=train_data,
                                             test_data=test_data,
                                             approaches=approaches,
                                             all_nodes=True).analyze() # or nodes_ids_to_analyze = [1,2,3,..]


NodeAnalysis
~~~~~~~~~~~~

.. code-block:: python

     from fedot.core.chains.chain import Chain
     from fedot.core.data.data import InputData
     from fedot.sensitivity.node_sensitivity import \
     NodeDeletionAnalyze, NodeAnalysis, NodeReplaceModelAnalyze

     chain = Chain()
     node_id_to_analyze = 2
     train_data = InputData.from_csv('train_file.csv')
     test_data = InputData.from_csv('test_file.csv')
     approaches = [NodeDeletionAnalyze, NodeReplaceModelAnalyze]

     node_result = NodeAnalysis(approaches=approaches). \
                analyze(chain=self.chain, node_id=node_id_to_analyze,
                        train_data=self.train_data,
                        test_data=self.test_data)

The evaluation result represents the ratio of origin quality score to score of the modified model.
After SA there will be a json file(default location at ``~home\Fedot\sensitivity``)
including the results per Node and detailed visualisation.

Result file example:

.. code-block:: json

     {
        "id = 0, model = knn": {
            "NodeDeletionAnalyze": 1.0,
            "NodeReplaceModelAnalyze": 0.9542077536133998
        },
        "id = 1, model = qda": {
            "NodeDeletionAnalyze": 1.0,
            "NodeReplaceModelAnalyze": 0.9970282317979198
        },
        "id = 2, model = xgboost": {
            "NodeDeletionAnalyze": 0.9836552748885586,
            "NodeReplaceModelAnalyze": 0.9766310955018235
        },
        "id = 3, model = knn": {
            "NodeDeletionAnalyze": 1.0312035661218424,
            "NodeReplaceModelAnalyze": 1.0068890990139132
        },
        "id = 4, model = logit": {
            "NodeDeletionAnalyze": 1.052005943536404,
            "NodeReplaceModelAnalyze": 0.9717682020802377
        },
        "id = 5, model = xgboost": {
            "NodeDeletionAnalyze": 0.9227340267459138,
            "NodeReplaceModelAnalyze": 0.965959746048899
        }
    }

Result visualization example:

What if to delete or replace every node in the chain

|DeletionAnalysis| |ReplacementAnalysis|

What if to replace original model(in red) to the certain one

|id_0| |id_1|

.. |DeletionAnalysis| image:: ../img/img_sensitivity/NodeDeletionAnalyze.jpg
   :width: 45%
.. |ReplacementAnalysis| image:: ../img/img_sensitivity/NodeReplaceModelAnalyze.jpg
   :width: 45%
.. |id_0| image:: ../img/img_sensitivity/knn_id_0_replacement.jpg
   :width: 45%
.. |id_1| image:: ../img/img_sensitivity/qda_id_1_replacement.jpg
   :width: 45%


.. _ChainStructureAnalysis: https://fedot.readthedocs.io/en/latest/api/sensitivity.html#fedot.sensitivity.chain_sensitivity.ChainStructureAnalyze
.. _NodeAnalysis: https://fedot.readthedocs.io/en/latest/api/sensitivity.html#fedot.sensitivity.node_sensitivity.NodeAnalysis
.. _NodeAnalyzeApproach: https://fedot.readthedocs.io/en/latest/api/sensitivity.html#fedot.sensitivity.node_sensitivity.NodeAnalyzeApproach