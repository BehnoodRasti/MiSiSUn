from dataclasses import dataclass


@dataclass(eq=False)
class BaseUnmixingModel:
    time: int = -1
    fitted: bool = False

    def register_time(self, time):
        self.fitted = True
        self.time = time

    @property
    def processing_time(self):
        if not self.fitted:
            return "Not fitted!"
        return f"{self} took {self.time:.2f}s"
