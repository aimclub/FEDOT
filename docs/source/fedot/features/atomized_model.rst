Using the chain as an atomic model
----------------------------------

To create more complex chains from other nested chains, we created a class
AtomizedModel, which wraps the chain in a wrapper and provides an interface
to this chain as to the usual Model class of our framework.

**Example of use:**

In order to add another Chain to the Chain as a Model, you need to wrap the
chain in the AtomizedModel class and all the functionality will be saved.

.. code-block:: python

    from fedot.core.chains.chain import Chain
    from fedot.core.models.atomized_model import AtomizedModel

    chain = Chain()
    nested_chain = Chain()
    atomized_model = AtomizedModel(nested_chain)
    chain.add_node(atomized_model)
