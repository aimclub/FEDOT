"""Tests for the experimental text detectors."""

import uuid

import pandas as pd
import pytest

from experiment.text_detection.detectors import (
    CurrentFedotDetector,
    Pr903Detector,
    TextDataDetectorV2,
)


@pytest.fixture
def detector():
    return TextDataDetectorV2()


@pytest.mark.parametrize('values, expected_reason', [
    (
        ['{TV, Wifi, Kitchen}', '{Washer, Heating, Kitchen}', '{Parking, Wifi, TV}'],
        'SERIALIZED_COLLECTION',
    ),
    (
        [
            'Armie Hammer, Annabelle Wallis, Tom Cullen',
            'Jake Gyllenhaal, Rene Russo, Bill Paxton',
            'Gerard Butler, Aaron Eckhart, Morgan Freeman',
        ],
        'PERSON_LIST',
    ),
    (
        ['a-new-summer-project', 'the-next-page-film', 'getting-boys-to-read'],
        'SLUG_VALUES',
    ),
    (
        ['Alice Brown', 'John Smith', 'Robert Green', 'Helen Davis'],
        'PERSON_NAMES',
    ),
    (
        ['https://site.org/1', 'https://site.org/2', 'https://site.org/3'],
        'STRUCTURED_URL_RATIO',
    ),
    (
        ['alex@site.org', 'bob@site.org', 'chris@site.org'],
        'STRUCTURED_EMAIL_RATIO',
    ),
    (
        ['/home/me/a.txt', '/home/me/b.txt', '/home/me/c.txt'],
        'STRUCTURED_PATH_RATIO',
    ),
    (
        ['100', '200', '300'],
        'NUMERIC_STRINGS',
    ),
])
def test_structured_values(detector, values, expected_reason):
    result = detector.analyze_column(pd.Series(values))
    assert result['is_text'] is False
    assert result['reason'] == expected_reason


def test_uuid_values(detector):
    values = pd.Series([str(uuid.uuid4()) for _ in range(5)])
    result = detector.analyze_column(values)
    assert result['is_text'] is False
    assert result['reason'] == 'STRUCTURED_UUID_RATIO'


def test_mixed_collection_values(detector):
    values = pd.Series([
        '{TV, Wifi}', '{Parking, Kitchen}', 'ordinary text',
        '["A", "B"]', 'another sentence',
    ])
    result = detector.analyze_column(values)
    assert result['reason'] == 'SERIALIZED_COLLECTION'


def test_long_repeated_description(detector):
    text = (
        'This product is comfortable and reliable for daily use. '
        'Customers appreciate the design and the durable material.'
    )
    values = pd.Series([text] * 50 + [text + ' Recommended.'] * 5)
    result = detector.analyze_column(values)
    assert result['is_text'] is True
    assert result['reason'] == 'LONG_TEXT'
    assert result['features']['unique_ratio'] < 0.1


def test_short_semantic_titles(detector):
    values = pd.Series([
        'A great choice for daily exercise',
        'Easy setup and excellent results',
        'The latest trends in modern design',
        'A practical guide to data analysis',
        'My experience with a new laptop',
        'How to improve your morning routine',
    ])
    result = detector.analyze_column(values)
    assert result['is_text'] is True
    assert result['reason'] == 'DIVERSE_SHORT_TEXT'


def test_result_matches_define_text_columns(detector):
    frame = pd.DataFrame({
        'comment': [
            'A good article about garden design',
            'An interesting guide to learning piano',
            'The best places to explore this summer',
            'A review of new machine learning tools',
        ],
        'email': ['a@site.org', 'b@site.org', 'c@site.org', 'd@site.org'],
    })
    assert detector.define_text_columns(frame) == ['comment']


@pytest.mark.parametrize('values, expected_reason', [
    ([None, '', ' ', None], 'TOO_FEW_VALUES'),
    (['only one', None, 'only two'], 'TOO_FEW_VALUES'),
    ([1, 2, 3, 4], 'NOT_STRING'),
    ([True, False, True, False], 'NOT_STRING'),
])
def test_invalid_or_sparse_values(detector, values, expected_reason):
    result = detector.analyze_column(pd.Series(values))
    assert result['is_text'] is False
    assert result['reason'] == expected_reason
    assert not result['features']


def test_nullable_string_column(detector):
    values = pd.Series([
        'The new book was an enjoyable read',
        None,
        'How I learned to play an instrument',
        'A useful overview of recent developments',
        'Planning a trip to the mountains',
    ], dtype='string')
    assert detector.analyze_column(values)['is_text'] is True


def test_person_list_helper(detector):
    assert detector._is_person_list('Alice Green, Robert White, Claire Smith')
    assert not detector._is_person_list('Apple, Lemon, Orange')
    assert not detector._is_person_list('Alice Green, Robert White')


def test_slug_helper(detector):
    assert detector._is_slug('this-is-a-slug')
    assert not detector._is_slug('This is a sentence')


def test_historical_baselines_do_not_modify_input():
    frame = pd.DataFrame({
        'title': [
            'A good article about design',
            'Why learning languages is useful',
            'An introduction to data analysis',
        ],
        'price': [10, 20, 30],
    })
    original = frame.copy(deep=True)
    for detector in (CurrentFedotDetector(), Pr903Detector()):
        assert isinstance(detector.define_text_columns(frame), list)
    pd.testing.assert_frame_equal(frame, original)
