import tempfile
import logging
from pathlib import Path
from assertpy import assert_that

from bennu_feature_extractor.environment_tools.env_files.env_file_pickle import EnvFilePickle

def test_pickle_read_write_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.pkl")
        logger.addHandler(logging.NullHandler())

        p = root / "obj.pkl"
        p.write_bytes(b"")
        f = EnvFilePickle(last_modified=None, actual_path=p, virtual_path=virtual_root / "obj.pkl", logger=logger)

        payload = {"answer": 42, "items": [1, 2, 3]}
        f.write(payload)
        read_back = f.read()
        assert_that(read_back).is_equal_to(payload)
