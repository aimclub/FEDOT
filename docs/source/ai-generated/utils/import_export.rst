Project Import and Export Functions
====================================

Functions for importing and exporting projects.

Functions:
    export_project_to_zip(): Convert pipeline to JSON, data to CSV, compress them to a zip archive, and save to the specified path.
    import_project_from_zip(): Unzips a zip file containing a project and returns the imported classes.

Args:
    zip_name (str): Absolute or relative path to the zip file with the exported project.
    pipeline: Pipeline object to export.
    train_data: Train InputData object to export.
    test_data: Test InputData object to export.
    opt_history: History of model optimization to export (if available).
    log_file_name: Name of the file with log to export.
