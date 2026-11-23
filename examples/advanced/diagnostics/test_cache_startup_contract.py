import pytest
from examples.advanced.diagnostics.cache_startup_contract import prepare_run


def test_prepare_is_exclusive_and_preserves_previous_artifacts(tmp_path):
    run=prepare_run(tmp_path,'prepared-false')
    assert (run/'cache').is_dir()
    artifact=run/'cache'/'preserved';artifact.write_text('keep')
    with pytest.raises(FileExistsError): prepare_run(tmp_path,'prepared-false')
    assert artifact.read_text()=='keep'


@pytest.mark.parametrize('name',['../escape','other','/absolute'])
def test_prepare_rejects_unknown_or_escaping_names(tmp_path,name):
    with pytest.raises(ValueError): prepare_run(tmp_path,name)


def test_prepare_rejects_missing_parent(tmp_path):
    with pytest.raises(ValueError): prepare_run(tmp_path/'missing','prepared-true')


def test_prepare_rejects_file_collision(tmp_path):
    p=tmp_path/'prepared-true';p.write_text('keep')
    with pytest.raises(FileExistsError): prepare_run(tmp_path,'prepared-true')
    assert p.read_text()=='keep'
