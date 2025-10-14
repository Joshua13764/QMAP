from abc import ABC, abstractmethod

class IVenvClient(ABC):
    @abstractmethod
    def test(self) -> None:
        pass