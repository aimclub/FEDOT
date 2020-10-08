import json
import os

from core.composer.chain import Chain
from utilities.synthetic.chain_template_new import ChainTemplate
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData
from cases.data.data_utils import get_scoring_case_data_paths
from utilities.synthetic.chain import chain_balanced_tree

CURRENT_PATH = str(os.path.dirname(__file__))


def create_chain() -> Chain:
    chain = Chain()
    node_logit = PrimaryNode('logit')

    node_lda = PrimaryNode('lda')
    node_lda.custom_params = {'n_components': 1}

    node_xgboost = PrimaryNode('xgboost')

    node_knn = PrimaryNode('knn')
    node_knn.custom_params = {'n_neighbors': 9}

    node_knn_second = SecondaryNode('knn')
    node_knn_second.custom_params = {'n_neighbors': 5}
    node_knn_second.nodes_from = [node_lda, node_knn]

    node_logit_second = SecondaryNode('logit')
    node_logit_second.nodes_from = [node_xgboost, node_lda]

    node_lda_second = SecondaryNode('lda')
    node_lda_second.custom_params = {'n_components': 1}
    node_lda_second.nodes_from = [node_logit_second, node_knn_second, node_logit]

    node_xgboost_second = SecondaryNode('xgboost')
    node_xgboost_second.nodes_from = [node_logit, node_logit_second, node_knn]

    node_knn_third = SecondaryNode('knn')
    node_knn_third.custom_params = {'n_neighbors': 8}
    node_knn_third.nodes_from = [node_lda_second, node_xgboost_second]

    chain.add_node(node_knn_third)

    return chain


def create_fitted_chain() -> Chain:
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)

    chain = create_chain()
    chain.fit(train_data)

    return chain


def create_json_models_files():
    chain = create_chain()
    chain_template = ChainTemplate(chain)
    # TODO нужно удалить папку, которая создается когда мы создаем шаблон цепочки из необученнух моделей
    unique_id = chain_template.unique_chain_id
    chain_template.export_to_json("test/data/test_chain_convert_to_json.json")

    chain_fitted = create_fitted_chain()
    chain_fitted.save_chain("test/data/test_fitted_chain_convert_to_json.json")

    chain_empty = Chain()
    chain_empty.save_chain("test/data/test_empty_chain_convert_to_json.json")

    os.rmdir(os.path.join(os.path.abspath('fitted_models'), unique_id))


def delete_json_models_files():
    def delete_models(chain):
        model_path = chain['nodes'][0]['trained_model_path']
        dir_path = os.path.dirname(os.path.abspath(model_path))
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        os.rmdir(dir_path)

    with open("test/data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        chain_fitted_object = json.load(json_file)
    with open("test/data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        chain_object = json.load(json_file)

    delete_models(chain_fitted_object)

    os.remove("test/data/test_fitted_chain_convert_to_json.json")
    os.remove("test/data/test_empty_chain_convert_to_json.json")
    os.remove("test/data/test_chain_convert_to_json.json")


create_json_models_files()


def test_chain_to_json_correctly():
    chain = create_chain()
    json_actual = chain.save_chain("test/data/1.json")

    with open("test/data/test_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    os.remove("test/data/1.json")
    assert json_actual == json.dumps(json_expected)


def test_chain_template_to_json_correctly():
    chain = create_chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.make_json()

    with open("test/data/test_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected)


def test_json_to_chain_correctly():
    chain = Chain()
    chain.load_chain("test/data/test_chain_convert_to_json.json")
    json_actual = chain.save_chain("test/data/1.json")

    chain_expected = create_chain()
    json_expected = chain_expected.save_chain("test/data/2.json")

    os.remove("test/data/1.json")
    os.remove("test/data/2.json")
    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_json_template_to_chain_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json("test/data/test_chain_convert_to_json.json")
    json_actual = chain_template.make_json()

    chain_expected = create_chain()
    chain_expected_template = ChainTemplate(chain_expected)
    json_expected = chain_expected_template.make_json()

    assert json.dumps(json_actual) == json.dumps(json_expected)


def test_fitted_chain_to_json_correctly():
    chain = Chain()
    chain.load_chain("test/data/test_fitted_chain_convert_to_json.json")
    json_actual = chain.save_chain("test/data/1.json")

    with open("test/data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    os.remove("test/data/1.json")
    assert json_actual == json.dumps(json_expected)


def test_fitted_chain_template_to_json_correctly():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    chain_template.import_from_json("test/data/test_fitted_chain_convert_to_json.json")
    json_actual = chain_template.make_json()

    with open("test/data/test_fitted_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected)


def test_empty_json_to_chain():
    chain = Chain()
    chain_template = ChainTemplate(chain)
    json_actual = chain_template.make_json()

    with open("test/data/test_empty_chain_convert_to_json.json", 'r') as json_file:
        json_expected = json.load(json_file)

    assert json_actual == json.dumps(json_expected)


def test_one_chain_object_save_load_vice_versa():
    chain_fitted_after = create_chain()
    chain_fitted_after.save_chain("test/data/1.json")

    chain_fitted = create_fitted_chain()
    json_first = chain_fitted.save_chain("test/data/2.json")

    chain_fitted_after.load_chain("test/data/2.json")
    json_second = chain_fitted_after.save_chain("test/data/3.json")

    for i in range(1, 4):
        os.remove(f"test/data/{i}.json")

    assert json_first == json_second


def test_absolute_relative_paths():
    chain = create_chain()
    chain.save_chain("test/data/1.json")

    absolute_path = os.path.join(os.path.abspath("test/data/2.json"))
    chain.save_chain(absolute_path)

    chain.load_chain("test/data/1.json")
    chain.load_chain(absolute_path)

    os.remove("test/data/1.json")
    os.remove(absolute_path)

    assert True


delete_json_models_files()






#
# def test_import_chain():
#     chain = Chain()
#     chain_fitted = create_fitted_chain()
#     chain_fitted.save_chain('test/data/my_chain.json')
#     chain.load_chain('/home/magleb/git/FEDOT/test/data/my_chain.json')
#     chain.save_chain('/home/magleb/git/FEDOT/test/data/my_chain_out.json')
#     print("SUCCESS")
#     assert True
#
# def test_fitted_chain_convert_to_json_correctly():
#     chain = create_fitted_chain()
#     # chain.save_chain('/home/magleb/git/FEDOT/test/data')
#     # chain.save_chain('test/data')
#     chain.save_chain('test/data/my_chain.json')
#     assert True

    # print(json_object_actual)
    # print()

    # with open(CURRENT_PATH + "/data/fitted_chain_to_json_test.json", 'r') as json_file:
    #     json_object_expected = json.load(json_file)

    # print(json_object_expected)

    # assert json_object_actual == json.dumps(json_object_expected)

# test_fitted_chain_convert_to_json_correctly()
# test_import_chain()
# test_chain_convert_to_json_correctly()
# test_fitted_chain_to_json_correctly()
# test_chain_to_json_correctly()
