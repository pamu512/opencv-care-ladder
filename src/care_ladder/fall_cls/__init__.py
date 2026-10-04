"""Kaggle fall-image classifier: license gate, split, train, ONNX export."""

from care_ladder.fall_cls.licenses import (
    EVAL_SLUG,
    EXCLUDED_SLUGS,
    SEARCH_URL,
    TRAIN_SLUG,
    DatasetCard,
    assert_train_allowed,
    classify_license,
)

__all__ = [
    "EVAL_SLUG",
    "EXCLUDED_SLUGS",
    "SEARCH_URL",
    "TRAIN_SLUG",
    "DatasetCard",
    "assert_train_allowed",
    "classify_license",
]
