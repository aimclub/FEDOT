Pipeline
========

Save/Load
=========

FEDOT provides methods for saving and loading pipelines in the ``Pipeline`` class:

- `save <https://github.com/aimclub/FEDOT/blob/master/fedot/core/pipelines/pipeline.py#L241>`_
    This method saves the pipeline to JSON representation with pickled fitted operations and preprocessing if there are any.

    There are five ways to save pipeline:
        - with ``create_subdir=True`` and ``is_datetime_in_path=True``
           An additional folder will be created inside the specified directory.
           The folder and name of JSON file with pipeline will contain timestamp.

        - with ``create_subdir=True`` and ``is_datetime_in_path=False``
           An additional folder will be created inside the specified directory.
           The folder and name of JSON file with pipeline will contain pipeline autoincrement index.
           The index is calculated based on the number of already saved pipelines in this directory.
           For example, if there are no saved pipelines in specified directory, than current pipeline
           will be saved as ``0_pipeline_saved.json`` in ``0_pipeline_saved`` subdir.
           This option is useful for pipelines with fitted_operations&preprocessing.

        - with ``create_subdir=False`` and ``is_datetime_in_path=True``
           Pipeline will be saved exactly in specified dir.
           The folder and name of JSON file with pipeline will contain timestamp.
           This option is useful for pipelines without fitted_operations&preprocessing.

        - with ``create_subdir=False`` and ``is_datetime_in_path=False``
           Pipeline will be saved exactly in specified dir.
           The name of JSON file with pipeline will be the same as the last folder in the path.
           For example, if ``C:\path\to\my\pipeline`` path was specified, than pipeline will be saved in
           ``C:\path\to\my\pipeline\pipeline.json``.
           This option is useful for pipelines without fitted_operations&preprocessing
           and when it's important to know the exact name of pipeline file.

        - with JSON file name in path
           For example, if path specified like this ``C:\path\to\my\pipeline\pipeline.json``,
           than pipeline will be saved exactly to this file. Fitted_operations&preprocessing will be saved in
           ``C:\path\to\my\pipeline\`` it there are any.
           Other args as ``create_subdir`` and ``is_datetime_in_path`` do not matter in this option.
           This option is useful for pipelines without fitted_operations&preprocessing
           and when it's important to know the exact name of pipeline file.


* `load <https://github.com/aimclub/FEDOT/blob/master/fedot/core/pipelines/pipeline.py#L263>`_


.. automodule:: fedot.core.pipelines.pipeline
   :members:
   :no-undoc-members:
