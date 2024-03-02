class CloudIndexDict(dict):
    def __init__(self, index_api):
        self.index_api = index_api
        super().__init__(self)
    def __missing__(self, key):
        self[key] = {int(k) : v for k, v in self.index_api.document(str(key)).get().to_dict().items()}
        return self[key]

class CloudIndex:
    def __init__(self, collection_) -> None:
        self.index_api = collection_
        self.cache = CloudIndexDict(self.index_api)
    def __getitem__(self, i: int|str):
        return self.cache[i]