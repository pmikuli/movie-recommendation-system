from pymilvus import MilvusClient

_client = None

def connect():
    global _client
    _client = MilvusClient(
            uri='http://root:password@127.0.0.1:19530',
            db_name='default'
        )

def create_collections():
    _client.create_collection(collection_name='users', dimension=128)
    _client.create_collection(collection_name='movies', dimension=128)
    _client.create_collection(collection_name='collaborative_filtering', dimension=128)

def insert_vector(collection, vector):
    noIdea = _client.insert(collection_name=collection, data=vector)
    print(noIdea)
    return noIdea

def query(collection):
    return _client.query(collection, '', limit=5)
