from typing import Any

from fedot.core.caching.hasher import Hasher
from fedot.core.caching.saver import Saver


class Cacher:
    def __init__(self):
        self.cache = {}

    def cache_data(self, data: Any):
        hash = Hasher.hash(data)
        if not is_in_index_db(hash):
            responce = Saver.save(data)
            index_db.add(hash, responce)
        else:
            responce = index_db.get(hash)
        return responce

    def get_data(self, data: Any):
        hash = Hasher.hash(data)
        if is_in_index_db(hash):
            responce = index_db.get(hash)
            data = Loader.load(responce)
            return data
        else:
            return None

    def is_in_index_db(self, hash: str):
        return hash in index_db

    def add_to_index_db(self, hash: str, responce: Any):
        index_db.add(hash, responce)

    def get_from_index_db(self, hash: str):
        return index_db.get(hash)