import two_tower_model.vectordatabase as vectordatabase

import random

dim = 128
vectors = [{ "id": i, "vector": [random.random() for _ in range(dim)]} for i in range(20)]

vectordatabase.connect()
vectordatabase.insert_vector('movies', vectors[0])
print(vectordatabase.query('movies'))