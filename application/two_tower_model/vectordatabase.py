import sys
print(sys.executable)

from pymilvus import MilvusClient

_client = None

def connect():
    global _client
    _client = MilvusClient(
            uri='http://root:password@127.0.0.1:19530',
            db_name='default'
        )

def create_collections():
    # _client.create_collection(collection_name='users', dimension=128)
    _client.create_collection(collection_name='movies', dimension=64)
    # _client.create_collection(collection_name='collaborative_filtering', dimension=128)

def insert_vector(collection, vector):
    noIdea = _client.insert(collection_name=collection, data=vector)
    return noIdea

def query(collection):
    return _client.query(collection, "id >= 0", limit=5)

def find_neighbors(collection, vector, k=10, filter_expression=None):
    search_params = {
        "metric_type": "COSINE",  # or "L2" if you use Euclidean
        "params": {"nprobe": 10}  # controls search scope
    }

    results = _client.search(
        collection_name=collection,
        data=vector,         # List of query vectors
        anns_field="vector",     # Field name in Milvus where vectors are stored
        search_params=search_params,
        limit=k,
        filter=filter_expression
    )

    return results