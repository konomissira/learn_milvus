import zipfile
import io
from PIL import Image
import open_clip  
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import torch

# Connexion à Milvus
connections.connect("default", host="localhost", port="19530")

# Chargement du modèle CLIP
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval()

# Schéma Milvus
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="prediction", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
]
schema = CollectionSchema(fields=fields, description="Fromages vectorisés avec CLIP")

# Création collection
collection_name = "fromages_clip"

# Vérification de l'existence de la collection
if utility.has_collection(collection_name):
    print(f"La collection '{collection_name}' existe déjà. Nous la supprimons.")
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)

# Lecture du .zip
zip_path = "C:\\fromages\\archive.zip"  # Chemin local vers le zip

with zipfile.ZipFile(zip_path, 'r') as archive:
    file_names = [f for f in archive.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    filenames, labels, embeddings = [], [], []
    counter = 1

    for name in file_names:
        # 🧷 Label = nom du dossier
        parts = name.split('/')
        if len(parts) < 2: continue  # Évite les erreurs sur dossiers vides
        label = parts[0].encode("cp437").decode("utf-8")
        filename = parts[-1].encode("cp437").decode("utf-8")

        # 📷 Chargement de l’image depuis le zip
        with archive.open(name) as file:
            img_bytes = file.read()
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image_input = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                filenames.append(filename)
                labels.append(label)
                embeddings.append(image_features.squeeze().tolist())
                counter += 1
                if counter == 100:
                    # Insertion dans Milvus
                    collection.insert([filenames, labels, [""] * len(filenames), embeddings])
                    filenames, labels, embeddings = [], [], []
                    counter = 1
                    print(f"⚙️ 100 images insérées")
                print(f"{name.encode("cp437").decode("utf-8")} traité")
            except Exception as X:
                print(f"⚠️ Erreur : {name}, {X}")


collection.flush()

# Index & load
collection.create_index("embedding", {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 200}
})
collection.load()

print(f"Collection '{collection_name}' : {collection.num_entities} images indexées.")