import networkx as nx
import pandas as pd

from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode


class StructureExplorer:
    """ Class for performing pipeline structure exploration. """

    def __init__(self):
        self.expected_paths_number = 0

    def check_structure_by_tag(self, pipeline, tag_to_check: str):
        nx_graph, node_labels = graph_structure_as_nx_graph(pipeline)

        # Assign information for all nodes in the graph
        nx_graph, info_df = self.enrich_with_information(nx_graph, node_labels)

        primary_df = info_df[info_df['node_type'] == 'primary']
        root_df = info_df[info_df['node_type'] == 'root']
        root_id = root_df.iloc[0]['node_id']
        for primary_id in primary_df['node_id']:
            for path in nx.all_simple_paths(nx_graph, source=primary_id, target=root_id):
                print(primary_id)
                print(path)

        # for primary_id in primary_df['node_id']:
        #     vertex_info = primary_df[primary_df['node_id'] == primary_id]
        #
        #     print(f'Start node: {str(vertex_info.iloc[0]["node_label"])}\n')
        #     vert_list = list(nx.bfs_successors(nx_graph, source=primary_id))
        #
        #     for component in vert_list:
        #         vertex = component[0]
        #         neighbors = component[1]
        #
        #         v_df = info_df[info_df['node_id'] == vertex]
        #         print(str(v_df.iloc[0]['node_label']))
        #
        #         for n in neighbors:
        #             v_df = info_df[info_df['node_id'] == n]
        #             print(str(v_df.iloc[0]['node_label']))
        #
        #         print('\n')

    def enrich_with_information(self, nx_graph, node_labels: dict):
        """
        Set additional information (operation name and node type) to nodes as attributes.
        There is also preparing pandas DataFrame with such information

        :param nx_graph: networkx object, graph
        :param node_labels: dictionary with ids and operation names
        """
        # Set names to nodes and additional info
        number_of_out_edges = nx_graph.degree._nodes
        number_of_in_edges = nx_graph.in_edges._adjdict

        info_df = []
        for node_id, node_label in node_labels.items():
            parent_numbers = len(number_of_in_edges[node_id])
            child_numbers = len(number_of_out_edges[node_id])

            if parent_numbers == 0:
                node_type = 'primary'
                self.expected_paths_number += child_numbers
            else:
                if child_numbers == 0:
                    # It is final node in the pipeline
                    node_type = 'root'
                else:
                    node_type = 'secondary'

            attrs = {node_id: {'operation': node_label, 'type': node_type}}
            nx.set_node_attributes(nx_graph, attrs)

            info_df.append([node_id, node_type, node_label, parent_numbers, child_numbers])

        info_df = pd.DataFrame(info_df, columns=['node_id', 'node_type', 'node_label',
                                                 'parent_number', 'child_number'])
        return nx_graph, info_df

    def primary_operation_search(self, pipeline, tag_to_check: str):
        """
        In the pipeline structure, a node with an operation with the appropriate tag is searched for.
        In this case the operations must have priority in the pipeline - in the PrimaryNode or not far from it.
        Validation performed only for basic places without combinations.
        Correct pipeline:
        operation with tag -> linear
        Incorrect pipeline:
        linear -> operation with tag

        :param pipeline: pipeline to check
        :param tag_to_check: search will be performed to find operation with such tag
        """

        are_primary_contain_right_nodes = []
        for current_node in pipeline.nodes:
            parent_nodes = current_node.nodes_from

            # Current node is secondary
            if type(current_node) is SecondaryNode:
                for parent_node in parent_nodes:
                    if type(parent_node) is PrimaryNode:
                        is_pos_good = self._operation_has_correct_position(current_node,
                                                                           parent_node,
                                                                           tag_to_check)
                        are_primary_contain_right_nodes.append(is_pos_good)

        has_operation = all(condition is True for condition in are_primary_contain_right_nodes)
        return has_operation

    @staticmethod
    def _operation_has_correct_position(current_node, parent_node, tag_to_check):
        # Operation in primary node is data source
        if 'data_source_table' in parent_node.operation.operation_type:
            # Check if current operation is needed operation
            if tag_to_check in current_node.tags:
                # Operation appear immediately after data source
                return True
            else:
                # Operation does not appear after data source
                return False

        # Otherwise - parent node (Primary) must contain wanted operation
        elif tag_to_check in parent_node.tags:
            return True
        else:
            # PrimaryNode also doesnt contain wanted operation
            return False
