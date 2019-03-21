from karabo_data import by_id, by_index


def test_slicing_reprs():
    ns = {'by_id': by_id, 'by_index': by_index}

    samples = [
        'by_id[:]',
        'by_id[:2]',
        'by_id[0:10:2]',
        'by_id[4::2, 7]',
        'by_index[:5, 3:12]',
        'by_index[-4:, ...]',
        'by_index[...]',
        'by_index[..., ::-1]',
    ]

    # These examples are canonically formatted, so their repr() should match
    for expr in samples:
        obj = eval(expr, ns)
        assert repr(obj) == expr
