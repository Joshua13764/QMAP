# tests/test_imports_simple.py
from importlib import import_module
from assertpy import assert_that

def test_modules_import():
    for name in ["bennu_feature_extractor"]:
        mod = import_module(name)
        assert_that(mod).is_not_none()
        assert_that(getattr(mod, "__spec__", None)).is_not_none()
