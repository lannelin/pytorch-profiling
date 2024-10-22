import time
from typing import Any


class SleepTransform:
    def __init__(self, sleep_time) -> None:
        self.sleep_time = sleep_time

    def __call__(self, inp: Any) -> Any:
        time.sleep(self.sleep_time)
        return inp
