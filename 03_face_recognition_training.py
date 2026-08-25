from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
import torch
import numpy as np
import json
from pathlib import Path
import cv2

images_path = Path("/home/finn/Arduino/ESP32_Cam/training images/Finn/00_all")

with open(images_path / "annotations.json", "r") as file:
    annotation_file = json.load(file)

img_name_dict = {img["image_id"]: img["image_name"] for img in annotation_file["images"]}

image_list = []
image_name_list = []

for i, ann in enumerate(annotation_file["annotations"]):
    x, y, w, h = ann["bbox"]
    image_name = img_name_dict[ann["image_id"]]
    image_list.append(
        cv2.imread(images_path / image_name)[y:y+h, x:x+w]
    )
    image_name_list.append(image_name)
    #cv2.imshow("face", image_list[i])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# vortrainiertes efficientnet_b1 nehmen und embeddings der vorletzten layer extrahieren
model = efficientnet_b1(weights = EfficientNet_B1_Weights.IMAGENET1K_V2)
# Ausgabe der layer "flatten" nehmen und in "embeddings" speichern
feature_extractor = create_feature_extractor(model, return_nodes = {"flatten": "embedding"}) 
feature_extractor.eval() # in evaluations modus setzen, da wir efficientnet nicht trainieren sondern nur benutzen

# jetzt müssen alle bilder auf 240x240 Pixel gebracht werden, dazu definition einer Funktion zum Transformieren von Bildern:
mean = [0.485, 0.456, 0.406] # werte aus offizieller Doku: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html#torchvision.models.EfficientNet_B1_Weights
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToPILImage(),          # OpenCV → PIL
    transforms.Resize((240, 240)),    # Bildgröße auf EfficientNet-Input
    transforms.ToTensor(),            # PIL → Tensor [0..1]
    transforms.Normalize(mean, std)
])

# jetzt preprocess auf alle Bilder anwenden, dann den crop ins Modell geben und aus der vorletzten Schicht extrahieren
embeddings = []
for image in image_list:
    prepped_image = preprocess(image)
    prepped_image = prepped_image.unsqueeze(0)
    with torch.no_grad():
        emb = feature_extractor(prepped_image)["embedding"].squeeze(0)
    embeddings.append(emb)
print(len(embeddings[1]))

# mit den Embeddings gibt es nun zwei Optionen. 
##### wird verwofen ########## 1. MLP mit 2 hidden Layers, Relu activation, am ende 2 knoten (Finn vs unbekannt) mit softmax activation, dropout in hidden layers
#    2. Average der Embeddings der Finn Bilder in Trainingsdaten berechnen und dann alles was innerhalb einer bestimmten Distanz liegt als Finn klassifizieren
#       + hier braucht man keine negative samples
#       + Euklidian distance und Cosine Similarity probieren, aber cosine wird vmtl besser sein
#       - da mehrere Bilder gemacht werden, werden bekannte Personen recht sicher erkannt, aber unbekannte nicht zuverlässig als solche erkannt
#          + da kann gelöst werden über eine Lücke in den Distanzen: Alles näher als A ist Finn, alles zwischen A und B nicht sicher, alles größer B ist unknown
#       - am besten dafür auch face tracking implementieren. Also Bilder sehr schnell hintereinander und dann faces im nächsten frame mit größter IoU ist gleiche Person

# Step 1 dafür ist, den Average der Embeddings ausrechnen 
embeddings_matrix = np.array(embeddings)
embeddings_mean = np.mean(embeddings_matrix, axis=0)

# Step 2 dafür ist, die Distanz aller Finn-Vektoren zu berechnen, um einen Radius für die Entscheidungsgrenzen festzulegen
#cos_sims = []
#for emb in embeddings:
#    cos_sim = np.dot(emb, embeddings_mean) / (np.linalg.norm(emb) * np.linalg.norm(embeddings_mean))
#    cos_sims.append(cos_sim)

#print(f"Cosine Simularities: {np.mean(cos_sims)}")
#for i in np.arange(len(image_list)):
#    print(f"{image_name_list[i]}: {cos_sims[i]}")
















images_path = Path("/home/finn/Arduino/ESP32_Cam/training images/Stella/00_all")

with open(images_path / "annotations.json", "r") as file:
    annotation_file = json.load(file)

img_name_dict = {img["image_id"]: img["image_name"] for img in annotation_file["images"]}

image_list = []
image_name_list = []

for i, ann in enumerate(annotation_file["annotations"]):
    x, y, w, h = ann["bbox"]
    image_name = img_name_dict[ann["image_id"]]
    image_list.append(
        cv2.imread(images_path / image_name)[y:y+h, x:x+w]
    )
    image_name_list.append(image_name)
    cv2.imshow("face", image_list[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# jetzt preprocess auf alle Bilder anwenden, dann den crop ins Modell geben und aus der vorletzten Schicht extrahieren
embeddings = []
for image in image_list:
    prepped_image = preprocess(image)
    prepped_image = prepped_image.unsqueeze(0)
    with torch.no_grad():
        emb = feature_extractor(prepped_image)["embedding"].squeeze(0)
    embeddings.append(emb)

cos_sims = []
for emb in embeddings:
    cos_sim = np.dot(emb, embeddings_mean) / (np.linalg.norm(emb) * np.linalg.norm(embeddings_mean))
    cos_sims.append(cos_sim)

print(f"Cosine Simularities: {np.mean(cos_sims)}")
for i in np.arange(len(image_list)):
    print(f"{image_name_list[i]}: {cos_sims[i]}")
