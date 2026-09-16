import logging as log
from collections import defaultdict


def init_logging():
    """Initialize logging."""
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class UniqueItemList:
    def __init__(self):
        self.items = []
        self.item_to_index = defaultdict(lambda: len(self.items))

    def index(self, item):
        idx = self.item_to_index[item]
        if idx == len(self.items):
            self.items.append(item)
        return idx
