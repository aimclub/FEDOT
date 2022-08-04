Example of project import and export
====================================

When you are working with the FEDOT framework, it becomes necessary to reproduce the created pipeline on another
computer with the available data. To do this, there are simple functions for exporting and importing **`project`**
. **'Project'** in this context means: pipeline, data, and logs.

Exporting a project
-------------------
Creating a pipeline using the composer. We set parameters for the composer and after the evolutionary algorithm works,
the composer returns the best generated pipeline.

To share a project or continue working with this thread on another computer, you can
use the function: **'export_project_to_zip'**:

.. code-block:: python

    from fedot.utilities.project_import_export import export_project_to_zip

    export_project_to_zip('project_name.zip', pipeline, train_data, test_data, 'project_name_log.log')

pipeline will be converted to JSON format, InputData objects to CSV format. The whole folder
will be compressed to zipfile: ``~/Fedot/projects/project_name.zip``.

Import project
--------------
To convert *FEDOT pipeline* back into *Python Object* and turn *CSV* into a *InputData Object*
that the framework is ready to work with, you need to call the function **'import_project_from_zip'**:

.. code-block:: python

    from sklearn.metrics import roc_auc_score as roc_auc
    from fedot.utilities.project_import_export import import_project_from_zip

    pipeline, train_data, test_data = import_project_from_zip('~/project_name.zip', verbose=True)

    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data)

    print(roc_auc(test_data.target, prediction.predict))



The function returns a ready-made *Python Object pipeline* and *InputData* and create unarchived folder in
default path: ``~/Fedot/projects/project_name`` with unarchived files.

Now you can upload, export, and share your project in ZIP format.
