import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from fedot.utilities.debug import FIT_ERROR_REPORT_DIR_ENV

FIT_ERROR_ALLOWLIST = {
    'builtins.TimeoutError': None,
    'fedot.core.data.merge.data_merger.DataMergeError': None,
    'numpy.linalg.LinAlgError': {'SVD did not converge'},
}


def _is_expected_fit_error(report: dict) -> bool:
    exception_type = report['exception_type']
    allowed_messages = FIT_ERROR_ALLOWLIST.get(exception_type, set())
    return allowed_messages is None or report['message'] in allowed_messages


def pytest_configure(config):
    report_dir = Path(tempfile.mkdtemp(prefix='fedot-fit-errors-'))
    config._fedot_fit_error_report_dir = report_dir
    config._fedot_previous_fit_error_report_dir = os.environ.get(FIT_ERROR_REPORT_DIR_ENV)
    os.environ[FIT_ERROR_REPORT_DIR_ENV] = str(report_dir)


def pytest_sessionfinish(session, exitstatus):
    report_dir = session.config._fedot_fit_error_report_dir
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in report_dir.glob('*.json')]
    if not reports:
        return

    expected = [report for report in reports if _is_expected_fit_error(report)]
    unexpected = [report for report in reports if not _is_expected_fit_error(report)]
    terminal_reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    if terminal_reporter is None:
        if unexpected:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        return

    terminal_reporter.write_sep('=', 'pipeline fit errors')

    if expected:
        error_counts = Counter((report['exception_type'], report['message']) for report in expected)
        terminal_reporter.write_line('Expected errors:')
        for (exception_type, message), count in error_counts.most_common():
            terminal_reporter.write_line(f'  {count} x {exception_type}: {message}')

    if unexpected:
        terminal_reporter.write_line('Unexpected errors:')
        for report in unexpected:
            terminal_reporter.write_line(
                f"\n{report['exception_type']}: {report['message']}\n"
                f"Pipeline: {report['graph_id']}\n{report['stack_trace']}"
            )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_unconfigure(config):
    previous_report_dir = config._fedot_previous_fit_error_report_dir
    if previous_report_dir is None:
        os.environ.pop(FIT_ERROR_REPORT_DIR_ENV, None)
    else:
        os.environ[FIT_ERROR_REPORT_DIR_ENV] = previous_report_dir
    shutil.rmtree(config._fedot_fit_error_report_dir, ignore_errors=True)
