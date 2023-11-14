Using the pipeline as an atomic model
-------------------------------------

To create more complex pipelines from other nested pipelines, we created a class
AtomizedModel, which wraps the pipeline in a wrapper and provides an interface
to this pipeline as to the usual Model class of our framework.

**Example of use:**

In order to add another Pipeline to the Pipeline as a Model, you need to wrap the
pipeline in the AtomizedModel class and all the functionality will be saved.

.. code-block:: python

    from fedot.core.pipelines.node import PipelineNode
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.operations.atomized_model import AtomizedModel

    pipeline = Pipeline()
    nested_pipeline = Pipeline(PipelineNode('ridge'))
    atomized_model = PipelineNode(AtomizedModel(nested_pipeline))
    pipeline.add_node(atomized_model)