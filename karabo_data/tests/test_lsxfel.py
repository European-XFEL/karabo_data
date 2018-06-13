from karabo_data import lsxfel
from karabo_data import H5File

def test_lsxfel_file(mock_lpd_data, capsys):
    with H5File(mock_lpd_data) as f:
        img_ds, index = lsxfel.find_image(f)
        assert img_ds.ndim == 4
        assert index['first'].shape == (480,)

    lsxfel.summarise_file(mock_lpd_data)
    out, err = capsys.readouterr()
    assert "480 trains, 128 frames/train" in out

def test_lsxfel_run(mock_fxe_run, capsys):
    lsxfel.summarise_run(mock_fxe_run)
    out, err = capsys.readouterr()

    assert "480 trains" in out
    assert "16 detector files" in out
