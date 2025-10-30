from abc import ABC
from typing import Tuple
import attr

from bennu_feature_extractor.environment_tools.base_classes.file_storage_persist_base import FileStoragePersistBase
from bennu_feature_extractor.environment_tools.file_storage_persists.runtime_only_persist import RuntimeOnlyPersist

@attr.define(frozen=True, slots=True)
class FSPathBase(ABC):
    path : Tuple[str]