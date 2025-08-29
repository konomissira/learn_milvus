import torch
import open_clip
from pymilvus import connections, Collection

# Chargement du modèle texte CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()


# Connexion à la collection Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("fromages_clip")
collection.load()

# Requête texte
# query = "fromage à pâte persillée"
query = "fromage de forme carrée"
text_inputs = tokenizer([query])
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Conversion en vecteur numpy
query_vector = text_features[0].cpu().numpy()

# Recherche dans Milvus
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"ef": 64}},
    limit=5,
    output_fields=["filename", "label"]
)

# Affichage des résultats
print(f"\n🔎 Résultats pour la requête : \"{query}\"")
for hit in results[0]:
    filename = hit.entity.get("filename")
    label = hit.entity.get("label")
    print(f"🧀 {filename} | Label : {label} | Score : {hit.score:.4f}")