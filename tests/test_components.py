import pytest

from project_name.apps import MyAppDROP
from project_name.data import  MyDataDROP

given = pytest.mark.parametrize


def test_myApp_class():
    assert MyAppDROP("a", "a").run() == "Hello from MyAppDROP"


def test_myData_class():
    assert MyDataDROP("a", "a").getIO() == "Hello from MyDataDROP"


def test_myData_dataURL():
    assert MyDataDROP("a", "a").dataURL == "Hello from the dataURL method"


