# inference.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import IntentVLM  # make sure this points to your model.py
import os

# ====== Paths ======
model_path = r"C:\Users\info\OneDrive\Desktop\3-1\iot\LightEMMA\drivelm\DriveLM\drivelm\intentvlm_best.pth"
image_path = r"C:\Users\info\Downloads\WhatsApp Image 2025-11-06 at 19.54.32_5d9549a8.jpg"

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Model ======
input_dim = 3 * 224 * 224
hidden_dim = 256
output_dim = 12  # 6 steps * 2 values
model = IntentVLM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ====== Preprocess Image ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).view(1, -1).to(device)  # flatten

# ====== Inference ======
with torch.no_grad():
    output = model(image_tensor)
    low_level_commands = output.view(6, 2).cpu().tolist()  # list of 6 (speed, curvature) pairs

# ====== Scene Description & High-Level Intent ======
# Placeholder: you can integrate a VLM or text description model here
#scene_description = "The image appears to be taken through a rearview mirror, showing the road behind the vehicle. There are multiple cars visible, and the background includes infrastructure like bridges. There's also a green road sign with directions. The image is somewhat blurry."
#high_level_intent = "The primary goal is to maintain a safe following distance from the vehicles ahead and stay within the current lane. Based on the road sign, the driver may also intend to take an exit."

# ====== Print Output in Desired Format ======
#print("*1. Scene Description*")
#print(scene_description)
#print("\n*2. High-Level Driving Intent*")
#print(high_level_intent)
print("\n*3. Low-Level Commands*")
print(low_level_commands)
# inference.py
# inference.py
# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# from model import IntentVLM
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # ====== Paths ======
# model_path = r"C:\Users\info\OneDrive\Desktop\3-1\iot\drivelm\DriveLM\drivelm\intentvlm_trained_with_blip.pth"
# image_path = r"C:\Users\info\OneDrive\Desktop\3-1\iot\drivelm\DriveLM\data\LightEmma\images\n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151610912404.jpg"

# # ====== Device ======
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ====== Load Model ======
# input_dim = 3 * 224 * 224
# hidden_dim = 256
# output_dim = 12
# model = IntentVLM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()

# # ====== BLIP for Scene Description ======
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-small")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-small").to(device)
# blip_model.eval()

# # ====== Preprocess Image ======
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# image = Image.open(image_path).convert("RGB")
# image_tensor = transform(image).view(1, -1).to(device)

# # ====== Inference ======
# with torch.no_grad():
#     low_level_output = model(image_tensor)
#     low_level_commands = low_level_output.view(6, 2).cpu().tolist()

# # Scene description with BLIP
# inputs = blip_processor(images=image, return_tensors="pt").to(device)
# with torch.no_grad():
#     output_ids = blip_model.generate(**inputs)
# scene_description = blip_processor.decode(output_ids[0], skip_special_tokens=True)

# # High-level intent (simple placeholder, could be enhanced)
# high_level_intent = "The primary goal is to navigate safely, follow traffic rules, and maintain appropriate speed."

# # ====== Print Outputs ======
# print("*1. Scene Description*")
# print(scene_description)
# print("\n*2. High-Level Driving Intent*")
# print(high_level_intent)
# print("\n*3. Low-Level Commands*")
# print(low_level_commands)
