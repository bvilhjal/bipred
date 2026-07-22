"""The lazy public API must remain loadable against the pinned ldpred3."""

import bipred


def test_all_lazy_public_exports_load():
    for name in bipred.__all__:
        assert getattr(bipred, name) is not None, name
