import networkx as nx
import pandas as pd
import numpy as np

from fedot.core.pipelines.convert import graph_structure_as_nx_graph

DEFAULT_SOURCE_NAME = 'default'


class PipelineStructureExplorer:
    """ Class for performing pipeline structure exploration.
    The class allows you to convert pipelines into a networkx graph and considers
    all possible paths from PrimaryNode (or PrimaryNodes) to root node. If at least
    one of the paths contains an invalid sequence of operations, the search performed
    by this class will detect it
    """

    _invariant_tags = {'encoding': 'categorical-ignore',
                       'imputation': 'nans-ignore'}

    def __init__(self):
        self.path_id = None
        self.graph = None
        self.paths = {}

    def check_structure_by_tag(self, pipeline, tag_to_check: str, source_name: str = DEFAULT_SOURCE_NAME):
        """
        In the pipeline structure, a node with an operation with the appropriate tag is searched for.
        In this case the operations must have priority in the pipeline - in the PrimaryNode or not far from it.

        Correct pipeline:
        operation with tag -> linear
        Incorrect pipeline:
        linear -> operation with tag

        :param pipeline: pipeline to check
        :param tag_to_check: find appropriate operation by desired tag
        (for example encoding or imputing)
        :param source_name: label of primary node for current input data. Set as 'default' if
        pipeline prepared for unimodal data
        """
        if len(pipeline.nodes) < 2:
            # Preprocessing needed for single-node pipeline
            return False
        self.path_id = 0
        self.graph, node_labels = graph_structure_as_nx_graph(pipeline)

        # Assign information for all nodes in the graph
        self.graph, info_df = self._enrich_with_information(node_labels)

        primary_df = info_df[info_df['node_type'] == 'primary']
        root_df = info_df[info_df['node_type'] == 'root']
        root_id = root_df.iloc[0]['node_id']
        for primary_id in primary_df['node_id']:
            if source_name == DEFAULT_SOURCE_NAME:
                # Check all possible paths in the pipeline
                for path in nx.all_simple_paths(self.graph, source=primary_id, target=root_id):
                    # For all paths perform checking if path (branch) has wanted operation
                    # in correct location or not
                    self.check_path(path, tag_to_check)
                    self.path_id += 1
            else:
                node_info = primary_df[primary_df['node_id'] == primary_id]
                node_name = node_info['node_label'].iloc[0].operation.operation_type
                if source_name == node_name:
                    # Only this path is going to be checked
                    for path in nx.all_simple_paths(self.graph, source=primary_id, target=root_id):
                        self.check_path(path, tag_to_check)
                        self.path_id += 1

        correct_branches = [branch['correctness'] for path_id, branch in self.paths.items()]
        if all(path_correctness is True for path_correctness in correct_branches):
            # All paths are correct
            return True
        else:
            # At least one branch in th graph cannot process desired type of data
            return False

    def check_path(self, path: list, tag_to_check: str):
        """
        Checking the path for operations take right places in the pipeline.

        :param path: path in the graph from PrimaryNode to root
        :param tag_to_check: find appropriate operation by desired tag
        """
        is_appropriate_operation, is_independent_operation = self._calculate_binary_paths(path, tag_to_check)

        # Define is branch correct or not
        is_branch_correct = self.is_current_branch_correct(is_appropriate_operation,
                                                           is_independent_operation,
                                                           len(path))

        self.paths[self.path_id].update({'correctness': is_branch_correct})
        return is_branch_correct

    def _enrich_with_information(self, node_labels: dict):
        """
        Set additional information (operation name and node type) to nodes as attributes.
        There is also preparing pandas DataFrame with such information

        :param node_labels: dictionary with ids and operation names
        """
        # Set names to nodes and additional info
        number_of_out_edges = self.graph.degree._nodes
        number_of_in_edges = self.graph.in_edges._adjdict

        info_df = []
        for node_id, node_label in node_labels.items():
            parent_numbers = len(number_of_in_edges[node_id])
            child_numbers = len(number_of_out_edges[node_id])

            if parent_numbers == 0:
                node_type = 'primary'
            else:
                if child_numbers == 0:
                    # It is final node in the pipeline
                    node_type = 'root'
                else:
                    node_type = 'secondary'

            attrs = {node_id: {'operation': node_label, 'type': node_type}}
            nx.set_node_attributes(self.graph, attrs)

            info_df.append([node_id, node_type, node_label, parent_numbers, child_numbers])

        info_df = pd.DataFrame(info_df, columns=['node_id', 'node_type', 'node_label',
                                                 'parent_number', 'child_number'])
        return self.graph, info_df

    def _calculate_binary_paths(self, path: list, tag_to_check: str) -> (list, list):
        """
        Calculate binary masks for considering path in the graph.
        For example, branch
        UID 584fef54 -> UID 568984edr45 -> UID 4566895ef13

        Can be decomposed as follows (if tag_to_check is encoding):

            * operations_path:
            imputation -> encoding -> ridge
            Listing the names of the operations in the nodes

            * is_appropriate_operation:
            False -> True -> False
            Whether there is an operation with a matching tag in the node or not

            * is_independent_operation
            True -> False -> False
            Are there any operations in the branch that can process data without
            errors, even if they have not been previously processed by encoding,
            e.g.
        """
        ignore_tag = self._invariant_tags.get(tag_to_check)

        # List with names of operations in the branch
        operations_path = []
        # Is the operation contain desired tag or not
        is_appropriate_operation = []
        # Can operation ignore errors when processing data without applying "desired operation"
        # For example: "desired operation" is imputing. If considering operation can process
        # data with nans in can_be_ignored list "True" will be added
        is_independent_operation = []
        for node_id in path:
            current_node = self.graph.nodes.get(node_id)

            node_tags = current_node['operation'].tags
            if tag_to_check in node_tags:
                is_appropriate_operation.append(True)
            else:
                # Operation in the node is not wanted one
                is_appropriate_operation.append(False)

                if ignore_tag is not None:
                    if ignore_tag in node_tags:
                        is_independent_operation.append(True)
                    else:
                        is_independent_operation.append(False)

            operations_path.append(current_node['operation'])

        self.paths.update({self.path_id: {'path': operations_path}})
        return is_appropriate_operation, is_independent_operation

    @staticmethod
    def is_current_branch_correct(is_appropriate_operation,
                                  is_independent_operation,
                                  path_len):
        """
        Based on predefined rules, the branch is checked for correctness.
        1) is_appropriate_operation
        2) is_independent_operation
        For example correct branch is:
            1) False -> True -> False
            2) True -> False -> False
            Data can be processed without errors due to wanted operation applied
            after independent operation
            True -> True -> after operation output everything is ok

        Incorrect branch is:
            1) False -> False -> True
            2) True -> False -> False
            Second operation is not wanted one and at the same time cannot
            process data without any transformation. So it will have an error
            True -> X False X -> True

        :param is_appropriate_operation: list with bool values is wanted operation placed in the node or not
        :param is_independent_operation: list with bool values is independent operation placed in the node or not
        :param path_len: length of list with graph nodes
        """

        # Check if operations were in the path
        is_appropriate_operation = np.array(is_appropriate_operation)
        # Number of True in list
        number_operations = is_appropriate_operation.sum()

        # True > 0 - so find ids with appropriate operations
        operation_ids = np.ravel(np.argwhere(is_appropriate_operation > 0))

        # Find independent operations in the path
        if len(is_independent_operation) > 0:
            is_independent_operation = np.array(is_independent_operation)
        else:
            is_independent_operation = np.array([False] * path_len)

        # Check independent operation presence in path
        number_independent_operations = is_independent_operation.sum()

        # By default it is incorrect
        is_branch_correct = False
        if number_operations == 0 and number_independent_operations == 0:
            # Path contain non independent operations and have no good ones
            is_branch_correct = False

        elif number_operations > 0 and number_independent_operations == 0:
            # Path contain good operations but have no independent
            if operation_ids[0] == 0:
                # Wanted operation in the first place
                is_branch_correct = True
            else:
                is_branch_correct = False

        elif number_operations == 0 and number_independent_operations > 0:
            # Have no wanted operations but have independent operations
            if number_independent_operations == path_len:
                # All operations are independent
                is_branch_correct = True
            else:
                is_branch_correct = False

        elif number_operations > 0 and number_independent_operations > 0:
            # Have both independent operation(s) and wanted operations
            path_before_good_operation = is_independent_operation[:operation_ids[0]]
            if path_before_good_operation.sum() == len(path_before_good_operation):
                # All operations are independent before wanted operation applying
                is_branch_correct = True
            else:
                is_branch_correct = False

        return is_branch_correct
