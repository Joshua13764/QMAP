import tempfile
import logging
from pathlib import Path
import pytest
from assertpy import assert_that

from bennu_feature_extractor.environment_tools.env_file_factory import EnvFileFactory
from bennu_feature_extractor.environment_tools.env_files.env_file_unsupported import EnvFileUnsupported

def test_unsupported_type_raises():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.unsupported")
        logger.addHandler(logging.NullHandler())

        p = root / "blob.bin"
        p.write_bytes(b"123")

        f = EnvFileFactory.create_env_file(file_path=p, virtual_path=virtual_root / "blob.bin", logger=logger)
        assert_that(f).is_instance_of(EnvFileUnsupported)

        with pytest.raises(NotImplementedError):
            f.read()
        with pytest.raises(NotImplementedError):
            f.write(b"x")
