from fedot.core.chains.chain import Chain
from fedot.core.composer.visualisation import ChainVisualiser

chain = Chain()
chain.load_chain('test\\data\\test_custom_json_template.json')

ChainVisualiser().visualise(chain)
