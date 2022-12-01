Pipeline save&load
==================

FEDOT provides methods for saving and loading pipelines in the ``Pipeline`` class:

- `save <https://github.com/aimclub/FEDOT/blob/master/fedot/core/pipelines/pipeline.py#L241>`_
    This method saves the pipeline to JSON representation with pickled fitted operations and preprocessing if there are any.

    There are five ways to save pipeline:
        1. **With** ``create_subdir=True`` **and** ``is_datetime_in_path=True``

           An additional folder will be created inside the specified directory.
           The folder and name of JSON file with pipeline will contain timestamp.

            **Example:**

            **NB.** *All examples assume that the folder where the pipelines are stored is initially empty.*

            .. code-block:: python

                problem = 'classification'
                train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'

                baseline_model = Fedot(problem=problem, timeout=1, seed=42)
                baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')

                pipeline = baseline_model.current_pipeline

                path_to_save = f'{fedot_project_root()}/saved_pipelines'

                pipeline.save(path=path_to_save, create_subdir=True, is_datetime_in_path=True)



            Directory with saved pipeline will look like this (with different timestamp):

            📦saved_pipelines

            ┣ 📂2022-11-16_15-53-49_pipeline_saved

            ┃ ┗ 📂fitted_operations

            ┃ ┗ 📂preprocessing

            ┃ ┗ 📜2022-11-16_15-53-49_pipeline_saved.json



        2. **With** ``create_subdir=True`` **and** ``is_datetime_in_path=False``

           An additional folder will be created inside the specified directory.
           The folder and name of JSON file with pipeline will contain pipeline autoincrement index.
           The index is calculated based on the number of already saved pipelines in this directory.

           For example, if there are no saved pipelines in specified directory, than current pipeline
           will be saved as ``0_pipeline_saved.json`` in ``0_pipeline_saved`` subdir.

           This option is useful for pipelines with fitted_operations&preprocessing.

            **Example:**

            .. code-block:: python

                problem = 'classification'
                train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'

                baseline_model = Fedot(problem=problem, timeout=1, seed=42)
                baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')

                pipeline = baseline_model.current_pipeline

                path_to_save = f'{fedot_project_root()}/saved_pipelines'

                pipeline.save(path=path_to_save, create_subdir=True, is_datetime_in_path=False)


            Directory with saved pipeline will look like this:

            📦saved_pipelines

            ┣ 📂0_pipeline_saved

            ┃ ┗ 📂fitted_operations

            ┃ ┗ 📂preprocessing

            ┃ ┗ 📜0_pipeline_saved.json


        3. **With** ``create_subdir=False`` **and** ``is_datetime_in_path=True``

           Pipeline will be saved exactly in specified dir.
           The folder and name of JSON file with pipeline will contain timestamp.

           This option is useful for pipelines without fitted_operations&preprocessing.

            **Example:**

            .. code-block:: python

                problem = 'classification'
                train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'

                baseline_model = Fedot(problem=problem, timeout=1, seed=42)
                baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')

                pipeline = baseline_model.current_pipeline

                path_to_save = f'{fedot_project_root()}/saved_pipelines'

                pipeline.save(path=path_to_save, create_subdir=False, is_datetime_in_path=True)


            Directory with saved pipeline will look like this:

            📦saved_pipeline

            ┣ 📂fitted_operations

            ┣ 📂preprocessing

            ┣ 📜2022-11-16_16-50-41_saved_pipeline.json

        4. **With** ``create_subdir=False`` **and** ``is_datetime_in_path=False``

           Pipeline will be saved exactly in specified dir.
           The name of JSON file with pipeline will be the same as the last folder in the path.

           For example, if ``C:\path\to\my\pipeline`` path was specified, than pipeline will be saved in
           ``C:\path\to\my\pipeline\pipeline.json``.

           This option is useful for pipelines without fitted_operations&preprocessing
           and when it's important to know the exact name of pipeline file.

            **Example:**

            .. code-block:: python

                problem = 'classification'
                train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'

                baseline_model = Fedot(problem=problem, timeout=1, seed=42)
                baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')

                pipeline = baseline_model.current_pipeline

                path_to_save = f'{fedot_project_root()}/saved_pipelines'

                pipeline.save(path=path_to_save, create_subdir=False, is_datetime_in_path=False)


            Directory with saved pipeline will look like this:

            📦saved_pipeline

            ┣ 📂fitted_operations

            ┣ 📂preprocessing

            ┣ 📜saved_pipeline.json

        5. **With JSON file name in path**

           For example, if path specified like this ``C:\path\to\my\pipeline\pipeline.json``,
           than pipeline will be saved exactly to this file. Fitted_operations&preprocessing will be saved in
           ``C:\path\to\my\pipeline\`` it there are any.

           Other args as ``create_subdir`` and ``is_datetime_in_path`` do not matter in this option.

           This option is useful for pipelines without fitted_operations&preprocessing
           and when it's important to know the exact name of pipeline file.

            **Example:**

            .. code-block:: python

                problem = 'classification'
                train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'

                baseline_model = Fedot(problem=problem, timeout=1, seed=42)
                baseline_model.fit(features=train_data_path, target='target', predefined_model='rf')

                pipeline = baseline_model.current_pipeline

                path_to_save = f'{fedot_project_root()}/saved_pipelines/best_pipeline.json'

                pipeline.save(path=path_to_save, create_subdir=True, is_datetime_in_path=False)


            Directory with saved pipeline will look like this:

            📦saved_pipeline

            ┣ 📂fitted_operations

            ┣ 📂preprocessing

            ┣ 📜best_pipeline.json


- `load <https://github.com/aimclub/FEDOT/blob/master/fedot/core/pipelines/pipeline.py#L263>`_
    Loads the pipeline ``JSON`` representation with pickled fitted operations.

    There two ways to load pipeline:
        1. To specify path to pipeline dir
            For example, if pipeline was saved to ``C:\FEDOT\saved\2022-11-16_15-53-49_pipeline_saved\2022-11-16_15-53-49_pipeline_saved.json``
            than path to load pipeline should be specified as ``C:\FEDOT\saved\2022-11-16_15-53-49_pipeline_saved``.

            Fitted_operations&preprocessing will be loaded automatically if there are any.

            **NB.** You can use the same path without modification to load pipeline only if it was saved in 3, 4 or 5 way.
            This is due to the fact that with such saving options it is known exactly in which folder JSON file with the pipeline was saved.

                **Example:**

                If the directory where needed pipeline is stored looks like this:

                📦saved_pipeline

                ┣ 📂fitted_operations

                ┣ 📂preprocessing

                ┣ 📜best_pipeline.json

                The pipeline can be loaded in the following way:

                .. code-block:: python

                    # path to dir with pipeline
                    path_to_load = f'{fedot_project_root()}/saved_pipeline'

                    pipeline2 = Pipeline().load(path_to_load)


        2. To specify path to JSON file with pipeline
            For example, if pipeline was saved to ``C:\FEDOT\saved\2022-11-16_15-53-49_pipeline_saved\2022-11-16_15-53-49_pipeline_saved.json``
            than path to load pipeline must be specified as ``C:\FEDOT\saved\2022-11-16_15-53-49_pipeline_saved\2022-11-16_15-53-49_pipeline_saved.json``.

            Fitted_operations&preprocessing will be loaded automatically if there are any.

                **Example:**

                If the directory where needed pipeline is stored looks like this:

                📦saved_pipeline

                ┣ 📂fitted_operations

                ┣ 📂preprocessing

                ┣ 📜best_pipeline.json

                The pipeline can be loaded in the following way:

                .. code-block:: python

                    # path to pipeline json
                    path_to_load = f'{fedot_project_root()}/saved_pipeline/best_pipeline.json'

                    pipeline2 = Pipeline().load(path_to_load)
