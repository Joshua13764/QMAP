from dataclasses import dataclass

import pandas as pd

from Boulder_Statistics.environment_tools.base_classes.fs_adapter_base import \
    FSAdapterBase
from Boulder_Statistics.environment_tools.fs_paths.fs_path_local_disk import \
    FSPathLocalDisk


@dataclass(frozen=True)
class FSPandasPickleAdapter(
        FSAdapterBase[pd.DataFrame, FSPathLocalDisk]):
    """Uses the pk module and pickle to save and load dataframes"""

    def read(self, path: FSPathLocalDisk) -> pd.DataFrame:
        return pd.read_pickle(path.actual_path.as_posix())

    def write(self, obj: pd.DataFrame,
              path: FSPathLocalDisk) -> None:
        obj.to_pickle(path.actual_path.as_posix())
