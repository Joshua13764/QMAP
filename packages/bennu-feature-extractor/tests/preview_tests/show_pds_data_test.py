from pathlib import Path
from bennu_feature_extractor.environment import Environment
from bennu_feature_extractor.environment_tools.env_file_base import EnvFileBase
from bennu_feature_extractor.environment_tools.env_files.env_file_pds4_xml import EnvFilePDS4XML

if __name__ == "__main__":
    src: Path = Path(r"C:\Users\Joshu\Documents\AO33_DATA\ocams_data_calibrated_detailed_survey\orex.ocams\data_calibrated\detailed_survey\20190225T111052S098_map_radL2pan.xml")
    file: EnvFileBase = EnvFilePDS4XML(actual_path=src, virtual_path=Path("/test.xml"), last_modified=None, logger=Environment.get_empty_environment().Logger)
    file.show()
