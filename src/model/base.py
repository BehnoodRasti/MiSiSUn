from dataclasses import dataclass


@dataclass
class BaseUnmixingModel:
    time: int = -1
    fitted: bool = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def register_time(self, time):
        self.fitted = True
        self.time = time

    @property
    def processing_time(self):
        if not self.fitted:
            return "Not fitted!"
        return f"{self} took {self.time:.2f}s"
