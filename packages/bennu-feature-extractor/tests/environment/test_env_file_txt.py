import tempfile
import logging
from pathlib import Path
from assertpy import assert_that

from bennu_feature_extractor.environment_tools.env_files.env_file_txt import EnvFileTxt

def test_txt_read_write_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.txt")
        logger.addHandler(logging.NullHandler())

        p = root / "sample.txt"
        p.write_text("")  # touch

        f = EnvFileTxt(last_modified=None, actual_path=p, virtual_path=virtual_root / "sample.txt", logger=logger)

        data = "hello world"
        f.write(data)
        # write() uses repr(); ensure content reflects that
        on_disk = p.read_text()
        assert_that(on_disk).contains("hello world").starts_with("'").ends_with("'")

        read_back = f.read()
        assert_that(read_back).is_equal_to(repr(data))

def test_txt_metadata_size_delete():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.txt-meta")
        logger.addHandler(logging.NullHandler())

        p = root / "meta.txt"
        p.write_text("initial")
        f = EnvFileTxt(last_modified=None, actual_path=p, virtual_path=virtual_root / "meta.txt", logger=logger)

        # Initially invalid (no last_modified)
        assert_that(f.check_metadata_valid()).is_false()

        lm = f.get_last_modified()
        assert_that(lm).is_type_of(float)

        assert_that(f.check_metadata_valid()).is_true()
        assert_that(f.get_size()).is_greater_than(0)

        f.delete()
        assert_that(p.exists()).is_false()
        try:
            f.get_size()
            assert False, "expected FileExistsError after delete"
        except FileExistsError:
            pass
