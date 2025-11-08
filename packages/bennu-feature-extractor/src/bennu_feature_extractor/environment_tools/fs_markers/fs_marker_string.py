import attr

from bennu_feature_extractor.environment_tools.base_classes.fs_marker_base import \
    FSMarkerBase


@attr.define(frozen=True, slots=True)
class FSMarkerString(FSMarkerBase):
    value: str

    def is_equivalent(self, target: FSMarkerBase) -> bool:
        if isinstance(target, 'FSMarkerString'):
            return target.value == self.value
        else:
            return False
