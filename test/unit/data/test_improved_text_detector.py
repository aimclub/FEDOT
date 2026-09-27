import uuid

import numpy as np
import pandas as pd
import pytest

from experiment.improved_text_detector import NewTextDataDetector


@pytest.fixture
def detector():
    return NewTextDataDetector()


def test_review_is_text(detector):
    column = pd.Series([
        "the product is very useful and works perfectly every day",
        "delivery was fast and the quality of the product is excellent",
        "I have been using this device for several months without problems",
        "the product is comfortable and very easy to use at home",
        "I would definitely recommend this product to other people",
    ])

    assert detector._column_contains_text(column) is True


def test_description_is_text(detector):
    column = pd.Series([
        "this article describes several methods for automatic text classification",
        "the paper presents a detailed comparison of modern machine learning models",
        "this study investigates different approaches to processing natural language data",
        "the article explains how neural networks can be applied to classification problems",
        "the authors compare several algorithms using multiple real world datasets",
    ])

    assert detector._column_contains_text(column) is True


def test_title_is_text(detector):
    column = pd.Series([
        "machine learning methods for text classification",
        "new approaches to neural network optimization",
        "the role of artificial intelligence in software development",
        "modern methods for time series forecasting",
        "automatic processing of text in machine learning systems",
    ])

    assert detector._column_contains_text(column) is True


def test_name_is_not_text(detector):
    column = pd.Series([
        "Alex Smith",
        "John Brown",
        "Anna White",
        "Mike Green",
        "Kate Black",
    ])

    assert detector._column_contains_text(column) is False


def test_product_name_is_not_text(detector):
    column = pd.Series([
        "Samsung Galaxy S25 Ultra",
        "Apple iPhone 17 Pro",
        "Sony WH1000XM6",
        "Asus Zenbook 14",
        "Lenovo ThinkPad X1",
    ])

    assert detector._column_contains_text(column) is False


def test_category_is_not_text(detector):
    column = pd.Series([
        "phone",
        "phone",
        "phone",
        "laptop",
        "laptop",
        "phone",
        "phone",
        "laptop",
        "phone",
        "laptop",
    ])

    assert detector._column_contains_text(column) is False


def test_url_is_not_text(detector):
    column = pd.Series([
        "https://example.com/product/1",
        "https://example.com/product/2",
        "https://example.com/product/3",
        "https://example.com/product/4",
        "https://example.com/product/5",
    ])

    assert detector._column_contains_text(column) is False


def test_email_is_not_text(detector):
    column = pd.Series([
        "alex@example.com",
        "john@example.com",
        "anna@example.com",
        "mike@example.com",
        "kate@example.com",
    ])

    assert detector._column_contains_text(column) is False


def test_uuid_is_not_text(detector):
    column = pd.Series([
        str(uuid.uuid4())
        for _ in range(10)
    ])

    assert detector._column_contains_text(column) is False


def test_product_id_is_not_text(detector):
    column = pd.Series([
        "PRD-001",
        "PRD-002",
        "PRD-003",
        "PRD-004",
        "PRD-005",
    ])

    assert detector._column_contains_text(column) is False


def test_file_path_is_not_text(detector):
    column = pd.Series([
        "/home/user/data/file_1.txt",
        "/home/user/data/file_2.txt",
        "/home/user/data/file_3.txt",
        "/home/user/data/file_4.txt",
        "/home/user/data/file_5.txt",
    ])

    assert detector._column_contains_text(column) is False


def test_numeric_strings_are_not_text(detector):
    column = pd.Series([
        "100",
        "200",
        "300",
        "400",
        "500",
    ])

    assert detector._column_contains_text(column) is False


def test_numeric_column_is_not_text(detector):
    column = pd.Series([
        10,
        20,
        30,
        40,
        50,
    ])

    assert detector._column_contains_text(column) is False


def test_empty_column_is_not_text(detector):
    column = pd.Series([
        None,
        None,
        np.nan,
        "",
        " ",
    ])

    assert detector._column_contains_text(column) is False


def test_column_with_nans_can_still_be_text(detector):
    column = pd.Series([
        "this product works very well and I use it every day",
        None,
        "the delivery was fast and everything arrived in good condition",
        np.nan,
        "this device is simple to use and works without any problems",
        "I would recommend this product because the overall quality is good",
    ])

    assert detector._column_contains_text(column) is True


def test_too_few_values_are_not_text(detector):
    column = pd.Series([
        "this is a long natural language sentence with several words",
        "another long natural language sentence with several different words",
    ])

    assert detector._column_contains_text(column) is False


def test_mixed_identifiers_are_not_text(detector):
    column = pd.Series([
        "ITEM-1001",
        "ITEM-1002",
        "ITEM-1003",
        "ITEM-1004",
        "ITEM-1005",
    ])

    assert detector._column_contains_text(column) is False


def test_define_text_columns(detector):
    dataframe = pd.DataFrame({
        "review": [
            "this product works very well and I use it every day",
            "the delivery was fast and the product arrived without damage",
            "the overall quality is excellent and I would buy it again",
            "this device is comfortable and very simple to use",
            "I recommend this product because it works exactly as expected",
        ],

        "name": [
            "Alex Smith",
            "John Brown",
            "Anna White",
            "Mike Green",
            "Kate Black",
        ],

        "product_id": [
            "PRD-001",
            "PRD-002",
            "PRD-003",
            "PRD-004",
            "PRD-005",
        ],

        "url": [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
            "https://example.com/4",
            "https://example.com/5",
        ],
    })

    result = detector.define_text_columns(
        dataframe
    )

    assert result == ["review"]