import pymongo

client = pymongo.MongoClient('mongodb://root:password@localhost:27017')
db = client['database']
collection = db['test']

document = {"test": "test"}
x = collection.insert_one(document)
print(x)