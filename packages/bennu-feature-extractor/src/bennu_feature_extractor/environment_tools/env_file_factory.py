from logging import Logger
from pathlib import Path

from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase

class EnvFileFactory:
    @staticmethod
    def create_env_file(file_path: Path, virtual_path : Path) -> EnvFileBase:

        match file_path:
            case Path(suffix=".txt"):
                from bennu_feature_extractor.environment_tools.env_files.env_file_txt import EnvFileTxt

                return EnvFileTxt(
                    actual_path_str=file_path.as_posix(),
                    virtual_path_str=virtual_path.as_posix(),
                )
            
            case Path(suffix=".pkl"):
                from bennu_feature_extractor.environment_tools.env_files.env_file_pickle import EnvFilePickle

                return EnvFilePickle(
                    actual_path_str=file_path.as_posix(),
                    virtual_path_str=virtual_path.as_posix(),
                )
            
            case Path(suffix=".xml"):
                from bennu_feature_extractor.environment_tools.env_files.env_file_pds4_xml import EnvFilePDS4XML

                return EnvFilePDS4XML(
                    actual_path_str=file_path.as_posix(),
                    virtual_path_str=virtual_path.as_posix(),
                )
            
            case Path(suffix=".png"):
                from bennu_feature_extractor.environment_tools.env_files.env_file_PNG import EnvFilePNG
                return EnvFilePNG(
                    actual_path_str=file_path.as_posix(),
                    virtual_path_str=virtual_path.as_posix(),
                )
            
            case Path(suffix = ".fits"):
                from bennu_feature_extractor.environment_tools.env_files.env_file_pds4_fits import EnvFilePDS4Fits

                return EnvFilePDS4Fits(
                    actual_path_str=file_path.as_posix(),
                    virtual_path_str=virtual_path.as_posix(),
                )

            case _:
                from bennu_feature_extractor.environment_tools.env_files.env_file_unsupported import EnvFileUnsupported

                return EnvFileUnsupported(
                    actual_path_str=file_path.as_posix(),
                    virtual_path_str=virtual_path.as_posix(),
                )