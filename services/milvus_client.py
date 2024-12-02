# milvus_client.py

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MilvusClient:
    def __init__(
        self, host='localhost', port='19530', collection_name='default_collection',
        embedding_dim=6, secure=False
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.connect(host, port, secure)
        self.create_collection()

    def connect(self, host, port, secure=False):
        try:
            connections.connect(
                alias='default',
                host=host,
                port=port,
                secure=secure
            )
            logger.info(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise e

    def create_collection(self):
        try:
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                    FieldSchema(name='metadata', dtype=DataType.JSON)
                ]
                schema = CollectionSchema(fields, description='Event collection')
                self.collection = Collection(name=self.collection_name, schema=schema)
                logger.info(f"Created collection '{self.collection_name}'")
            else:
                self.collection = Collection(name=self.collection_name)
                logger.info(f"Using existing collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to create or load collection: {e}")
            raise e

    def insert(self, vectors, metadatas):
        # Ensure vectors are in the correct format (list of float32)
        # Convert vectors to numpy arrays of type float32
        vectors = [np.array(vector, dtype=np.float32) for vector in vectors]

        # Construct entities as a list of dictionaries
        entities = [
            {
                "vector": vector.tolist(),  # Convert numpy array back to list
                "metadata": metadata
            }
            for vector, metadata in zip(vectors, metadatas)
        ]

        # Insert data into Milvus
        self.collection.insert(entities)

        # Flush the collection to ensure data is persisted
        self.collection.flush()

        # Build index if not exists
        if not self.collection.has_index():
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            logger.info("Creating index...")
            self.collection.create_index(field_name="vector", index_params=index_params)
            logger.info("Index created successfully.")

        # Load the collection to make data available for querying
        logger.info("Loading collection...")
        self.collection.load()
        logger.info("Collection loaded successfully.")

    def close(self):
        connections.disconnect('default')
