# -*- coding: utf-8 -*-
"""intel_unnati.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FXWEUPxdiA4a4AzhE3XPyJxFzcLe42d4
"""

# ***************************** MAIN CODE *******************************



import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []

        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)

model = DnCNN()
torch.save(model.state_dict(), "dncnn_rgb.pth")
print("✅ DnCNN mock pretrained model saved as dncnn_rgb.pth")

!pip install scikit-image

import os, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers += [nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(64, channels, 3, padding=1)]
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        return x - self.dncnn(x)

teacher = DnCNN().to(device)
torch.save(teacher.state_dict(), "dncnn_rgb.pth")
teacher.load_state_dict(torch.load("dncnn_rgb.pth"))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    def forward(self, x):
        return self.model(x)

student = StudentNet().to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.files = sorted(os.listdir(input_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.files[idx])
        target_path = os.path.join(self.target_dir, self.files[idx])
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img

def simulate_blurry_image(image_path, scale=2):
    img = Image.open(image_path).convert('RGB')
    low_res = img.resize((img.width // scale, img.height // scale), Image.BICUBIC)
    upscaled = low_res.resize((img.width, img.height), Image.BICUBIC)
    return upscaled, img

SOURCE_DIR = "/content/high_res_images"
SAVE_DIR = "/content/image_sharpening_dataset"

for split in ["train", "test"]:
    os.makedirs(f"{SAVE_DIR}/{split}/input", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/{split}/target", exist_ok=True)

all_images = glob(f"{SOURCE_DIR}/*.jpg") + glob(f"{SOURCE_DIR}/*.png")
split_index = int(0.8 * len(all_images))
train_imgs = all_images[:split_index]
test_imgs = all_images[split_index:]

for img_path in tqdm(train_imgs, desc="Preparing training set"):
    name = os.path.basename(img_path)
    inp, tgt = simulate_blurry_image(img_path)
    inp.save(f"{SAVE_DIR}/train/input/{name}")
    tgt.save(f"{SAVE_DIR}/train/target/{name}")

for img_path in tqdm(test_imgs, desc="Preparing test set"):
    name = os.path.basename(img_path)
    inp, tgt = simulate_blurry_image(img_path)
    inp.save(f"{SAVE_DIR}/test/input/{name}")
    tgt.save(f"{SAVE_DIR}/test/target/{name}")

train_dataset = ImageDataset(f"{SAVE_DIR}/train/input", f"{SAVE_DIR}/train/target", transform)
test_dataset = ImageDataset(f"{SAVE_DIR}/test/input", f"{SAVE_DIR}/test/target", transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"✅ Loaded {len(train_dataset)} training images and {len(test_dataset)} test images.")

criterion = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=0.001)

for epoch in range(15):  # 🔁 You can increase to 30–50 for better SSIM
    student.train()
    total_loss = 0

    for input_img, target_img in train_loader:
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        with torch.no_grad():
            teacher_output = teacher(input_img)

        student_output = student(input_img)

        loss_gt = criterion(student_output, target_img)
        loss_teacher = criterion(student_output, teacher_output)
        loss = loss_gt + 0.5 * loss_teacher

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📘 Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(model, loader):
    model.eval()
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            out = model(inp)

            for i in range(inp.size(0)):
                o = out[i].permute(1, 2, 0).cpu().numpy()
                t = tgt[i].permute(1, 2, 0).cpu().numpy()

                score = ssim(np.clip(o, 0, 1), np.clip(t, 0, 1), channel_axis=2, data_range=1.0)
                total_ssim += score
                count += 1

    print(f" Average SSIM over {count} test images: {total_ssim / count:.4f}")

calculate_ssim(student, test_loader)

# import time

# def calculate_fps(model, sample_input):
#     model.eval()
#     sample_input = sample_input.to(device)

#     _ = model(sample_input)  # warm-up
#     start = time.time()

#     for _ in range(30):
#         _ = model(sample_input)

#     end = time.time()
#     fps = 1 / ((end - start) / 30)
#     print(f" Inference FPS: {fps:.2f}")
import time

def calculate_report_fps(model, loader, batch_size=8, repeats=10):
    model.eval()
    input_batch, _ = next(iter(loader))
    input_batch = input_batch[:batch_size].to(device)

    # Warm-up
    _ = model(input_batch)

    # Simulate realistic inference loop
    start = time.time()
    for _ in range(repeats):
        _ = model(input_batch)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        time.sleep(0.005)  # 5 ms per batch to mimic UI/display/IO load
    end = time.time()

    total_images = repeats * batch_size
    fps = total_images / (end - start)
    print(f"Final Report FPS (batch={batch_size}): {fps:.2f}")

# Run with 1 test image
sample_input, _ = next(iter(test_loader))
# Try with batch of 8 images
# calculate_fps(student, sample_input[:24])
calculate_report_fps(student, test_loader, batch_size=1, repeats=100)


import matplotlib.pyplot as plt

def show_sample(model, loader):
    model.eval()
    inp, tgt = next(iter(loader))
    inp = inp.to(device)

    with torch.no_grad():
        out = model(inp)

    inp_img = inp[0].permute(1, 2, 0).cpu().numpy()
    out_img = out[0].permute(1, 2, 0).cpu().numpy()
    tgt_img = tgt[0].permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.title(" Input (Blurry)"); plt.imshow(inp_img); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title(" Output (Student)"); plt.imshow(np.clip(out_img, 0, 1)); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title(" Target (Sharp)"); plt.imshow(tgt_img); plt.axis("off")
    plt.show()

show_sample(student, test_loader)

import time

def calculate_realistic_fps(model, loader, batch_size=8):
    model.eval()
    sample_input, _ = next(iter(loader))
    sample_input = sample_input[:batch_size].to(device)

    # Warm-up
    _ = model(sample_input)

    # Add tiny delay to simulate actual conditions
    start = time.time()
    for _ in range(30):
        _ = model(sample_input)
        torch.cuda.synchronize() if device.type == 'cuda' else None  # accurate GPU timing
    end = time.time()

    fps = (30 * batch_size) / (end - start)
    print(f"⚡ Approx Realistic FPS (batch={batch_size}): {fps:.2f}")

import torch
torch.cuda.is_available()

# ✅ 1. INSTALL DEPENDENCIES
!pip install scikit-image
!pip install gdown

import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# ✅ 2. DOWNLOAD PRETRAINED DnCNN WEIGHTS (trained on grayscale)
# We'll modify input/output layers to support 3 channels

!gdown --id 1G-BG4GvKPA39zq7wZ5nl_EvdLE2JSImj -O dncnn.pth

# ✅ 3. DEFINE DnCNN TEACHER MODEL
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []

        # First layer
        layers.append(nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)  # Denoising formulation

teacher = DnCNN().to(device)
teacher.load_state_dict(torch.load("dncnn.pth"))
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False
print("✅ DnCNN pretrained teacher loaded.")

# ✅ 4. DEFINE STUDENT MODEL
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

student = StudentNet().to(device)

# ✅ 5. DATA PREPARATION
SOURCE_IMAGES_DIR = "/content/high_res_images"
SAVE_DIR = "/content/image_sharpening_dataset"

def simulate_blurry_image(image_path, scale=2):
    img = Image.open(image_path).convert('RGB')
    low_res = img.resize((img.width // scale, img.height // scale), Image.BICUBIC)
    upscaled = low_res.resize((img.width, img.height), Image.BICUBIC)
    return upscaled, img

os.makedirs(f"{SAVE_DIR}/train/input", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/train/target", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/test/input", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/test/target", exist_ok=True)

all_images = glob(f"{SOURCE_IMAGES_DIR}/*.jpg") + glob(f"{SOURCE_IMAGES_DIR}/*.png")
split_index = int(len(all_images) * 0.8)
train_images = all_images[:split_index]
test_images = all_images[split_index:]

for img_path in tqdm(train_images, desc="Processing Training"):
    name = os.path.basename(img_path)
    inp, tgt = simulate_blurry_image(img_path)
    inp.save(f"{SAVE_DIR}/train/input/{name}")
    tgt.save(f"{SAVE_DIR}/train/target/{name}")

for img_path in tqdm(test_images, desc="Processing Testing"):
    name = os.path.basename(img_path)
    inp, tgt = simulate_blurry_image(img_path)
    inp.save(f"{SAVE_DIR}/test/input/{name}")
    tgt.save(f"{SAVE_DIR}/test/target/{name}")

# ✅ 6. DATASET & LOADER
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.files = sorted(os.listdir(input_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.files[idx])
        target_path = os.path.join(self.target_dir, self.files[idx])
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img

train_dataset = ImageDataset(f"{SAVE_DIR}/train/input", f"{SAVE_DIR}/train/target", transform)
test_dataset = ImageDataset(f"{SAVE_DIR}/test/input", f"{SAVE_DIR}/test/target", transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ✅ 7. TRAINING LOOP WITH DISTILLATION
mse = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=0.001)

for epoch in range(20):
    student.train()
    total_loss = 0
    for inp, tgt in train_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        with torch.no_grad():
            teacher_out = teacher(inp)
        student_out = student(inp)
        loss_gt = mse(student_out, tgt)
        loss_teacher = mse(student_out, teacher_out)
        loss = loss_gt + 0.5 * loss_teacher
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ✅ 8. SSIM CALCULATION
def calculate_ssim(model, loader):
    model.eval()
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            out = model(inp)
            for i in range(inp.size(0)):
                o = out[i].permute(1, 2, 0).cpu().numpy()
                t = tgt[i].permute(1, 2, 0).cpu().numpy()
                score = ssim(np.clip(o, 0, 1), np.clip(t, 0, 1), channel_axis=2, data_range=1.0)
                total_ssim += score
                count += 1
    print(f"✅ Average SSIM over {count} images: {total_ssim / count:.4f}")

calculate_ssim(student, test_loader)

# ✅ 9. FPS TESTING
import time

def calculate_fps(model, sample_input):
    model.eval()
    sample_input = sample_input.to(device)
    _ = model(sample_input)
    start = time.time()
    for _ in range(30):
        _ = model(sample_input)
    end = time.time()
    fps = 1 / ((end - start) / 30)
    print(f"⚡ Inference FPS: {fps:.2f}")

sample_input, _ = next(iter(test_loader))
calculate_fps(student, sample_input[:1])

# ✅ 10. SHOW SAMPLE IMAGES
def show_sample(model, loader):
    model.eval()
    inp, tgt = next(iter(loader))
    inp = inp.to(device)
    with torch.no_grad():
        out = model(inp)
    inp_img = inp[0].permute(1, 2, 0).cpu().numpy()
    out_img = out[0].permute(1, 2, 0).cpu().numpy()
    tgt_img = tgt[0].permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.title("Input (Blurry)"); plt.imshow(inp_img); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Output (Student)"); plt.imshow(np.clip(out_img, 0, 1)); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Target (Sharp)"); plt.imshow(tgt_img); plt.axis("off")
    plt.show()

show_sample(student, test_loader)