"""Regression tests for automatic text-column detection."""

import uuid

import numpy as np
import pandas as pd
import pytest

from fedot.core.data.data_detection import TextDataDetector


@pytest.fixture
def detector():
    return TextDataDetector()


@pytest.mark.parametrize('values', [
    ['Alex Smith', 'John Brown', 'Anna White', 'Mike Green'],
    [
        'Armie Hammer, Annabelle Wallis, Tom Cullen',
        'Jake Gyllenhaal, Rene Russo, Bill Paxton',
        'Gerard Butler, Aaron Eckhart, Morgan Freeman',
    ],
    [
        '{TV, Wifi, Kitchen}',
        '{Parking, Heating, Washer}',
        '{Kitchen, Wifi, TV}',
    ],
    [
        'the-next-page-feature-film',
        'getting-boys-to-read-quick-tips',
        'building-a-house-in-winter',
    ],
    [
        'https://example.org/first',
        'https://example.org/second',
        'https://example.org/third',
    ],
    [
        'alice@example.org',
        'bob@example.org',
        'carol@example.org',
    ],
    ['ITEM-001', 'ITEM-002', 'ITEM-003'],
    ['12', '13', '14', '15'],
    ['/tmp/data/file1.csv', '/tmp/data/file2.csv', '/tmp/data/file3.csv'],
])
def test_structured_values_are_not_text(detector, values):
    assert detector._column_contains_text(pd.Series(values)) is False


def test_uuid_is_not_text(detector):
    values = pd.Series([str(uuid.uuid4()) for _ in range(5)])
    assert detector._column_contains_text(values) is False


def test_repeated_description_is_text(detector):
    description = (
        'This is a comfortable jacket made of cotton. '
        'It is suitable for everyday use and easy to wash.'
    )
    values = pd.Series([description] * 60 + [description + ' Nice fit.'] * 3)

    result = detector.analyze_column(values)

    assert result['is_text'] is True
    assert result['features']['unique_ratio'] < 0.1


def test_short_titles_are_text(detector):
    values = pd.Series([
        'How to choose your first camera',
        'A useful introduction to machine learning',
        'Five tips for studying at home',
        'A review of the new laptop model',
        'Why good sleep matters for learning',
    ])

    assert detector._column_contains_text(values) is True


def test_product_titles_are_text(detector):
    values = pd.Series([
        'Lightweight running shoes for women',
        'Comfortable cotton jacket with a hood',
        'Waterproof travel backpack with pockets',
        'Soft wool sweater for cold weather',
        'Durable bicycle helmet with ventilation',
    ])

    assert detector._column_contains_text(values) is True


def test_nullable_string_series(detector):
    values = pd.Series([
        'How I learned to play the piano',
        None,
        'An introduction to healthy cooking',
        'A useful guide to exploring the city',
        'My experience with a new computer',
    ], dtype='string')

    assert detector._column_contains_text(values) is True


@pytest.mark.parametrize('values', [
    pd.Series([None, None, np.nan, '']),
    pd.Series(['one text', None, 'two texts']),
    pd.Series([10, 20, 30, 40]),
    pd.Series([True, False, True, False]),
])
def test_invalid_columns_are_not_text(detector, values):
    assert detector._column_contains_text(values) is False


def test_column_names_do_not_affect_detection(detector):
    values = [
        'A review of an interesting new book',
        'Simple steps to start learning Python',
        'How to prepare for the next interview',
        'A practical guide to growing flowers',
    ]
    dataframe = pd.DataFrame({
        'text': values,
        'unrelated_name': values,
        'identifier': ['ID-1', 'ID-2', 'ID-3', 'ID-4'],
    })

    assert detector.define_text_columns(dataframe) == [
        'text', 'unrelated_name',
    ]


def test_detector_keeps_dataframe_unchanged(detector):
    dataframe = pd.DataFrame({
        'title': [
            'A good guide to learning French',
            'How to choose a new bicycle',
            'My review of a useful device',
        ],
        'category': ['books', 'sports', 'electronics'],
    })
    original = dataframe.copy(deep=True)

    detector.define_text_columns(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_find_sparse_columns(detector):
    dataframe = pd.DataFrame({
        'description': ['Useful details', None, None, None, None, None,
                        None, None, None, None, None],
        'category': ['a'] * 11,
    })

    assert detector.find_sparse_columns(dataframe) == ['description']


def test_csv_import_detects_text_automatically(tmp_path):
    dataframe = pd.DataFrame({
        'review': [
            'A useful product that arrived very quickly',
            'The package was intact and everything worked',
            'This was easy to install and simple to use',
            'Excellent quality with a comfortable design',
            'My family enjoyed using this every day',
        ],
        'quantity': [2, 1, 3, 4, 2],
        'target': [1, 0, 1, 1, 0],
    })
    path = tmp_path / 'reviews.csv'
    dataframe.to_csv(path, index=False)

    MultiModalData = pytest.importorskip(
        'fedot.core.data.multi_modal'
    ).MultiModalData
    result = MultiModalData.from_csv(
        file_path=str(path),
        target_columns='target',
        index_col=None,
    )

    assert 'data_source_text/review' in result
    assert 'data_source_table' in result
    assert len(result['data_source_text/review'].features) == len(dataframe)


def test_csv_import_does_not_use_target_as_text(tmp_path):
    dataframe = pd.DataFrame({
        'value': [1, 2, 3, 4],
        'target': [
            'This is a long natural-language description',
            'Another interesting description about the product',
            'Here is a useful review of the product',
            'A long review containing some meaningful words',
        ],
    })
    path = tmp_path / 'targets.csv'
    dataframe.to_csv(path, index=False)

    MultiModalData = pytest.importorskip(
        'fedot.core.data.multi_modal'
    ).MultiModalData
    result = MultiModalData.from_csv(
        file_path=str(path),
        target_columns='target',
        index_col=None,
    )

    assert not any(key.startswith('data_source_text/') for key in result)


@pytest.mark.parametrize('url', [
    'https://example.com',
    'http://example.com/products/123',
    'www.example.com',
    'www.example.org/articles?page=2',
    'example.com/path',
    'example.org/products/123',
])
def test_url_formats(detector, url):
    column = pd.Series([url] * 5)

    assert detector._column_contains_links(column)
    assert not detector._column_contains_text(column)


@pytest.mark.parametrize('value', [
    'report.pdf',
    'dataset.csv',
    'image.jpeg',
    'version 1.2.3',
    'some.example.text',
])
def test_non_url_values(detector, value):
    column = pd.Series([value] * 5)

    assert not detector._column_contains_links(column)


def test_mixed_url_formats(detector):
    dataframe = pd.DataFrame({
        'links': [
            'https://example.com',
            'www.example.org',
            'example.com/path',
            'http://example.net/article',
            'some ordinary text',
        ],
    })

    assert detector.find_link_columns(dataframe) == ['links']


@pytest.mark.parametrize('values', [
    [
        'José García, Иван Иванов, François Dupont',
        'Émile Durand, Сергей Петров, María López',
        'André Martin, Анна Смирнова, João Silva',
    ],
    [
        'Tom Cruise, Brad Pitt, Leonardo DiCaprio',
        'Emma Stone, Ryan Gosling, Jennifer Lawrence',
        'Morgan Freeman, Robert Downey, Chris Evans',
    ],
])
def test_person_lists(detector, values):
    column = pd.Series(values)

    result = detector.analyze_column(column)

    assert not result['is_text']
    assert result['reason'] == 'PERSON_LIST'


def test_title_case_is_not_rejected_as_person_names(detector):
    column = pd.Series([
        'Machine Learning Methods',
        'Modern Network Architectures',
        'Deep Neural Optimization',
        'Distributed Software Design',
        'Statistical Language Processing',
    ])

    result = detector.analyze_column(column)

    assert result['is_text']


@pytest.mark.parametrize('size', [5, 25, 100])
def test_repeated_descriptions(detector, size):
    descriptions = [
        'This product is comfortable and works very well every day',
        'The delivery was fast and the overall quality is excellent',
        'I have used this device for several months without problems',
        'The installation was simple and the instructions were clear',
        'I would recommend this product to friends and family',
    ]

    values = [
        descriptions[index % len(descriptions)]
        for index in range(size)
    ]

    assert detector._column_contains_text(pd.Series(values))


@pytest.mark.parametrize('values', [
    [
        'This product works reliably and is easy to use every day',
        'The package arrived quickly and everything was in good shape',
        'I would recommend this device because it performs very well',
        'The instructions were clear and the installation was simple',
        'Customer support answered my questions and solved the issue',
    ],
    [
        'Этот продукт работает отлично и его удобно использовать',
        'Доставка прошла быстро и товар пришел в хорошем состоянии',
        'Новая модель показывает хорошие результаты обработки текста',
        'Система правильно определяет признаки в описаниях товаров',
        'Алгоритм помогает находить полезную информацию в отзывах',
    ],
])
def test_text_in_different_languages(detector, values):
    assert detector._column_contains_text(pd.Series(values))


def test_serialized_collections(detector):
    column = pd.Series([
        '{TV, Wifi, Kitchen, Parking}',
        '{Heating, Washer, Dryer, Internet}',
        '{Kitchen, TV, Air conditioning, Wifi}',
        '{Parking, Heating, Washer, Dryer}',
        '{Wifi, Kitchen, Internet, Television}',
    ])

    result = detector.analyze_column(column)

    assert not result['is_text']
    assert result['reason'] == 'SERIALIZED_COLLECTION'


def test_structured_values_do_not_depend_on_column_name(detector):
    values = [
        'https://example.com/1',
        'www.example.com',
        'example.com/path',
        'http://example.org/page',
        'https://example.net/5',
    ]

    dataframe = pd.DataFrame({
        'description': values,
        'url': values,
    })

    assert detector.define_text_columns(dataframe) == []
