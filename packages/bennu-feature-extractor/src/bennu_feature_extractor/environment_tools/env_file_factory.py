from logging import Logger
from pathlib import Path

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class EnvFileFactory:
    @staticmethod
    def create_env_file(file_path: Path, virtual_path : Path, logger : Logger) -> EnvFileBase:

        match file_path:
            case Path(suffix=".txt"):
                from bennu_feature_extractor.environment_tools.env_files.env_file_txt import EnvFileTxt

                return EnvFileTxt(
                    last_modified=None,
                    actual_path=file_path,
                    virtual_path=virtual_path,
                    logger = logger
                )
            
            case Path(suffix=".pkl"):
                from bennu_feature_extractor.environment_tools.env_files.env_file_pickle import EnvFilePickle

                return EnvFilePickle(
                    last_modified=None,
                    actual_path=file_path,
                    virtual_path=virtual_path,
                    logger = logger
                )

            case _:
                from bennu_feature_extractor.environment_tools.env_files.env_file_unsupported import EnvFileUnsupported

                return EnvFileUnsupported(
                    last_modified=None,
                    actual_path=file_path,
                    virtual_path=virtual_path,
                    logger = logger
                )