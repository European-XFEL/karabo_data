from karabo_data import lsxfel


def test_lsxfel_file(mock_lpd_data, capsys):
    lsxfel.summarise_file(mock_lpd_data)
    out, err = capsys.readouterr()
    assert "480 trains, 128 frames/train" in out


def test_lsxfel_run(mock_fxe_run, capsys):
    lsxfel.summarise_run(mock_fxe_run)
    out, err = capsys.readouterr()

    assert "480 trains" in out
    assert "16 detector files" in out
