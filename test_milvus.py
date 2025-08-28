from pymilvus import connections

# connect to Milvus running on localhost:19530
connections.connect("default", host="127.0.0.1", port="19530")

print("✅ Connected to Milvus successfully!")
