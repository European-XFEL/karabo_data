from karabo_data import RunDirectory
from karabo_data import validation


def test_validate_run(mock_fxe_run):
    rv = validation.RunValidator(mock_fxe_run)
    rv.validate()
