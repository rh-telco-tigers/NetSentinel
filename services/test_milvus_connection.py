from pymilvus import connections, Collection, utility

try:
    connections.connect(
        alias='default',
        host='localhost',
        port='19530',
        secure=False
    )
    print("Connected to Milvus successfully.")

    collections = utility.list_collections()
    print(f"Available collections: {collections}")

    for collection_name in collections:
        collection = Collection(name=collection_name)
        collection.flush()
        print(f"Collection: {collection_name}, Schema: {collection.schema}")
        print(f"Number of entities in collection: {collection.num_entities}")
        # Load the collection into memory
        collection.load()
        # Retrieve some data
        results = collection.query(expr="id >= 0", output_fields=["id", "metadata"])
        print(f"Retrieved records: {results}")

    connections.disconnect('default')

except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
