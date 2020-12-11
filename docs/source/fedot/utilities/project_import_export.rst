Example of project import and export
===================================================================

When you are working with the FEDOT framework, it becomes necessary to reproduce the created chain on another
computer with the available data. To do this, there are simple functions for exporting and importing **`project'**
. **'Project'** on that context means: chain, data, and logs.

Exporting a project
-----------------------
Creating a chain using the composer. We set parameters for the composer and after the evolutionary algorithm works,
the composer returns the best generated chain.

.. code-block:: python

    from cases.data.data_utils import get_scoring_case_data_paths
    from fedot.core.models.data import InputData
    from fedot.core.composer.gp_composer.fixed_structure_composer import FixedStructureComposerBuilder
    from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements
    from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    from test.chain.test_chain_tuning import get_class_chain

    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    available_model_types = ['logit', 'lda', 'knn']

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 pop_size=2, num_of_generations=1,
                                 crossover_prob=0.4, mutation_prob=0.5,
                                 add_single_model_chains=False)

    reference_chain = get_class_chain()
    builder = FixedStructureComposerBuilder(task=Task(TaskTypesEnum.classification)).with_initial_chain(
        reference_chain).with_metrics(metric_function).with_requirements(req)
    composer = builder.build()

    chain = composer.compose_chain(data=train_data)

To share a project or continue working with this thread on another computer, you can
use the function: **'export_project_to_zip'**:

.. code-block:: python

    from fedot.utilities.project_import_export import export_project_to_zip

    # export_project_to_zip(chain: Chain, train_data: InputData, test_data: InputData,
    #                       zip_name: str, log_file_name: str = None, verbose: bool = False):

    export_project_to_zip(chain, train_data, test_data,
                          'project_name.zip', 'project_name_log.log', verbose=True)

Chain will be converted to JSON format, InputData objects to CSV format. And all folder
will be compressed to zipfile: ``home/user/Fedot/projects/zip_name.zip``.

Import project
-----------------------
To convert *FEDOT Chain* back into *Python Object* and turn *CSV* into a *InputData Object*
that the framework is ready to work with, you need to call the function **'import_project_from_zip'**:

.. code-block:: python

    from sklearn.metrics import roc_auc_score as roc_auc
    from fedot.utilities.project_import_export import import_project_from_zip

    # import_project_from_zip(zip_path: str, verbose: bool = False) -> [Chain, InputData, InputData]:
    chain, train_data, test_data = import_project_from_zip('/home/user/downloads/project_name.zip', verbose=True)

    chain.fit(train_data)
    prediction = chain.predict(test_data)

    print(roc_auc(test_data.target, prediction.predict))



The function returns a ready-made *Python Object Chain* and *InputData* and create unarchived folder in
default path: ``/home/user/Fedot/projects/project_name`` with unarchived files.

Now you can upload, export, and share your project in ZIP format.