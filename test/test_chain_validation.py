import pytest

from core.chain_validation import (
    has_no_cycle,
    has_primary_nodes,
    has_no_self_cycled_nodes,
    has_no_isolated_nodes,
    validate,
    has_no_isolated_components,
)
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import LogRegression

ERROR_PREFIX = 'Invalid chain configuration:'


def test_chain_has_cycles():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])
    y2.nodes_from.append(y4)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    with pytest.raises(Exception) as exc:
        assert has_no_cycle(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has cycles'


def test_chain_has_no_cycles():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    assert has_no_cycle(chain)


def test_has_isolated_nodes():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    with pytest.raises(ValueError) as exc:
        assert has_no_isolated_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has isolated nodes'


def test_multi_root_chain():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    with pytest.raises(Exception) as exc:
        assert chain.root_node
    assert str(exc.value) == f'{ERROR_PREFIX} More than 1 root_nodes in chain'


def test_has_primary_ndoes():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    chain.add_node(y1)
    chain.add_node(y2)
    assert has_primary_nodes(chain)


def test_has_no_primary_nodes():
    chain = Chain()
    y1 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=None)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    chain.add_node(y1)
    chain.add_node(y2)
    with pytest.raises(Exception) as exc:
        assert has_primary_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain does not have primary nodes'


def test_has_no_self_cycled_nodes():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y2.nodes_from.append(y2)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    with pytest.raises(Exception) as exc:
        assert has_no_self_cycled_nodes(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has self-cycled nodes'


def test_validate_chain():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y3 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y4 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1, y2])
    y6 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3, y4])
    y7 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y5, y6])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    chain.add_node(y6)
    chain.add_node(y7)
    validate(chain)


def test_has_no_isolated_components():
    chain = Chain()
    y1 = NodeGenerator.primary_node(input_data=None, model=LogRegression())
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[])
    y5 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y4])
    y6 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y5])
    y4.nodes_from.append(y4)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    chain.add_node(y6)
    with pytest.raises(Exception) as exc:
        assert has_no_isolated_components(chain)
    assert str(exc.value) == f'{ERROR_PREFIX} Chain has isolated components'
