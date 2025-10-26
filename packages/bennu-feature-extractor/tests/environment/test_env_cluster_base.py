import tempfile
import logging
from pathlib import Path
from assertpy import assert_that

from bennu_feature_extractor.environment_tools.env_cluster_base import EnvCluster
from bennu_feature_extractor.environment_tools.env_files.env_file_txt import EnvFileTxt
from bennu_feature_extractor.environment_tools.env_files.env_file_pickle import EnvFilePickle

def _make_files(root: Path, virtual_root: Path, logger):
    p1 = root / "a.txt"; p1.write_text("hello")
    p2 = root / "b.pkl"; p2.write_bytes(b"")  # touch
    f1 = EnvFileTxt(last_modified=None, actual_path=p1, virtual_path=virtual_root / "a.txt", logger=logger)
    f2 = EnvFilePickle(last_modified=None, actual_path=p2, virtual_path=virtual_root / "b.pkl", logger=logger)
    f2.write({"k": "v"})
    return f1, f2

def test_cluster_total_size_and_check_metadata():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.cluster")
        logger.addHandler(logging.NullHandler())

        f1, f2 = _make_files(root, virtual_root, logger)
        cluster = EnvCluster(files=[f1, f2], logger=logger)

        # prime last_modified and then verify metadata validity
        for f in cluster.files:
            _ = f.get_last_modified()
        assert_that(cluster.check_metadata()).is_true()

        size_sum = sum(f.get_size() for f in [f1, f2])
        assert_that(cluster.get_total_size()).is_equal_to(size_sum)

def test_cluster_delete_removes_all():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.cluster-del")
        logger.addHandler(logging.NullHandler())

        f1, f2 = _make_files(root, virtual_root, logger)
        cluster = EnvCluster(files=[f1, f2], logger=logger)
        cluster.delete()

        assert_that(f1.actual_path.exists()).is_false()
        assert_that(f2.actual_path.exists()).is_false()

def test_cluster_pickle_roundtrip_strips_and_rehydrates_logger():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        virtual_root = root / "virtual"
        logger = logging.getLogger("bennu-tests.cluster-pickle")
        logger.addHandler(logging.NullHandler())

        f1, f2 = _make_files(root, virtual_root, logger)
        cluster = EnvCluster(files=[f1, f2], logger=logger)

        data = cluster.get_pickle_repr()
        assert_that(data).is_instance_of((bytes, bytearray))

        loaded = EnvCluster.from_pickle_repr(data, logger=logger)
        assert_that(loaded.logger).is_equal_to(logger)
        assert_that([f.logger for f in loaded.files]).contains_only(logger)
