import os

import pytest
import unittest


from chiles_daliuge.NewChiliesSplit import fetch_original_ms

SKIP_REMOTE_DATA_TESTS = True

def test_cwd():
    assert os.getcwd() == 'tests'

@pytest.fixture
def setup_environment():
    os.environ['DLG_ROOT'] = f"{os.getcwd()}/dlg"
    yield os.environ['DLG_ROOT']
    os.unlink(os.environ["DLG_ROOT"])

@pytest.mark.skipif(SKIP_REMOTE_DATA_TESTS,
                    reason="This test downloads a large amount of data.")
def test_fetch_original_ms():
    response = input("Please let us know you are okay running the tests")
    os.environ['DLG_ROOT'] = os.getcwd()
    name_list = fetch_original_ms(
        source_dir="acacia-chiles:2025-04-chiles01/originals_short/",
                      year_list=["2013-2014", "2015"],
                      copy_directory="$DLG_ROOT/MeasurementSets",METADATA_DB='test', add_to_db=False)


