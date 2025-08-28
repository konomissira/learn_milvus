from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient
from sentence_transformers import SentenceTransformer
import re

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Chargement du modèle d'embedding
model = SentenceTransformer('all-MiniLM-L6-v2')  # Dimension de 384

# Lecture du fichier texte
with open("pg55860.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Nettoyage et découpage en phrases
# sentences = re.split(r'(?<=[.!?])\s+', text.strip())
sentences = sent_tokenize(text, language='french')
sentences = [re.sub(r'(\s+|\n)', ' ', s).strip() for s in sentences]
# sentences = [s.replace('\n', '') for s in sentences]

print(f"Nombre de phrases à encoder : {len(sentences)}")
print(sentences[:5])  # pour voir un aperçu

# Génération des embeddings
embeddings = model.encode(sentences, convert_to_numpy=True)

# Affichage de la taille des embeddings
print(f"Taille des embeddings : {embeddings.shape}")

# Connexion à Milvus
connections.connect("default", host="localhost", port="19530")

# Définition du schéma de la collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000)
]
schema = CollectionSchema(fields, description="Embeddings des phrases de Balzac")

# Création de la collection
collection_name = "balzac_sentences"

# Vérification de l'existence de la collection
if utility.has_collection(collection_name):
    print(f"La collection '{collection_name}' existe déjà. Nous la supprimons.")
    utility.drop_collection("balzac_sentences")

collection = Collection(name=collection_name, schema=schema)

# Insertion des données
data = [
    {"embedding": emb.tolist(), "text": sent}
    for emb, sent in zip(embeddings, sentences)
]

print(f"Nombre d'éléments à insérer : {len(data)}")

insert_result = collection.insert(data)
print("IDs insérés :", insert_result.primary_keys[:5])
collection.flush()  # Assure la persistance des données

# Affichage du nombre d'éléments insérés
print(f"Nombre d'éléments insérés : {collection.num_entities}")

# Création de l'index
collection.create_index(field_name="embedding", index_params={
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
})

# Chargement de la collection en mémoire
collection.load()

# Récupération du nombre d'entités
print(f"Nombre d'entités : {collection.num_entities}")

# Récupération des statistiques de la collection
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
stats = client.get_collection_stats(collection_name="balzac_sentences")

# Affichage du nombre de lignes
print(f"Nombre de lignes : {stats['row_count']}")

chercher = "Une réflexion misanthrope sur la société et la nature humaine"
# Encodage de la phrase de recherche
query_embedding = model.encode([chercher], convert_to_numpy=True)

# Requête de recherche
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 200}
}

results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=5,
    expr=None,
    output_fields=["text"]
)
# Affichage des résultats
for result in results[0]:
    text = result.entity.get('text', '') if hasattr(result, 'entity') and result.entity else ''
    print(f"ID : {result.id}, distance : {result.distance}, texte : {text}")