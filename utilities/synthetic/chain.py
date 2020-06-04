from core.composer.chain import Chain


def chain_with_fixed_structure() -> Chain:
    """
    Generates chain with a fixed structure of nodes and links
    :return:
    """
    raise NotImplementedError()


def chain_with_random_links() -> Chain:
    """
    Generates chain with a fixed nodes structure but with random links
    :return:
    """
    raise NotImplementedError()


def chain_full_random() -> Chain:
    """
    Generates chain with random amount of nodes and links
    :return:
    """
    raise NotImplementedError()


def chain_balanced_tree() -> Chain:
    """
    Generates chain with balanced tree-like structure
    :return:
    """
    raise NotImplementedError()
