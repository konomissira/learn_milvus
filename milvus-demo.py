from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# 1) Connect
connections.connect("default", host="127.0.0.1", port="19530")

# 2) Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=4),
]
schema = CollectionSchema(fields, description="Demo collection")

# 3) Recreate collection cleanly
collection_name = "demo_collection"
if collection_name in utility.list_collections():
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)

# 4) Insert sample data
vectors = [
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6, 0.7],
    [0.9, 0.8, 0.7, 0.6],
]
collection.insert([vectors])

# 5) Build index with explicit metric (match search metric!)
index_params = {
    "index_type": "AUTOINDEX",
    "metric_type": "L2",
    "params": {}
}
collection.create_index(field_name="vector", index_params=index_params)

# 6) Load to memory
collection.load()

# 7) Search with the SAME metric
query = [[0.1, 0.2, 0.3, 0.4]]
res = collection.search(
    data=query,
    anns_field="vector",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=2,
)

print("🔎 Search results:")
for hit in res[0]:
    print(f"ID: {hit.id}, distance: {hit.distance}")
