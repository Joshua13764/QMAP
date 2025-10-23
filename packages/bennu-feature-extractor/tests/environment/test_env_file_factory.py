import tempfile
import logging
from pathlib import Path
from assertpy import assert_that

from bennu_feature_extractor.environment_tools.env_file_factory import EnvFileFactory
from bennu_feature_extractor.environment_tools.env_files.env_file_txt import EnvFileTxt
from bennu_feature_extractor.environment_tools.env_files.env_file_pickle import EnvFilePickle
from bennu_feature_extractor.environment_tools.env_files.env_file_unsupported import EnvFileUnsupported

def test_factory_dispatch_by_suffix():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.factory")
        logger.addHandler(logging.NullHandler())

        txt = root / "note.txt"; txt.write_text("x")
        pkl = root / "data.pkl"; pkl.write_bytes(b"")
        other = root / "blob.bin"; other.write_bytes(b"")

        f_txt = EnvFileFactory.create_env_file(file_path=txt, virtual_path=virtual_root / "note.txt", logger=logger)
        f_pkl = EnvFileFactory.create_env_file(file_path=pkl, virtual_path=virtual_root / "data.pkl", logger=logger)
        f_other = EnvFileFactory.create_env_file(file_path=other, virtual_path=virtual_root / "blob.bin", logger=logger)

        assert_that(f_txt).is_instance_of(EnvFileTxt)
        assert_that(f_pkl).is_instance_of(EnvFilePickle)
        assert_that(f_other).is_instance_of(EnvFileUnsupported)
