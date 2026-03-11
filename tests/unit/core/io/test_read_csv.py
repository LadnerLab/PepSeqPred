import pytest
from pepseqpred.core.io.read import parse_int_csv, parse_float_csv

pytestmark = pytest.mark.unit


def test_parse_int_csv_success():
    assert parse_int_csv("11,22,33", "--split-seeds") == [11, 22, 33]


def test_parse_int_csv_invalid():
    with pytest.raises(ValueError, match="CSV list of integers"):
        parse_int_csv("11,abc", "--split-seeds")


def test_parse_float_csv_success():
    assert parse_float_csv("0.1,1,2.5", "--dropouts") == [0.1, 1.0, 2.5]


def test_parse_float_csv_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_float_csv(" , ", "--dropouts")
