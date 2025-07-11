# Dog_Skin_Disease

# 🐾 Pawsitive Care - Canine Dermatology Scanner

An AI-powered web app to detect common dog skin diseases from uploaded images and provide preliminary treatment guidance.

## 📌 Overview

Dogs often suffer from skin conditions like allergic dermatitis, fungal infections, or mange. This project uses a deep learning model (ResNet18) to analyze uploaded photos and predict the likely skin disease. Pet owners can use this tool to get early insights — but always consult a veterinarian for a professional diagnosis!

## 🚀 Features

✅ Upload multiple photos of a dog's skin condition  
✅ Instant AI-based prediction with confidence scores  
✅ Shows likely condition, severity, description, symptoms, and suggested treatments  
✅ Clean, responsive web UI built with **Streamlit**  
✅ Includes disclaimers and next-step recommendations

---

## 🧩 How It Works

1. The app loads a **pre-trained ResNet18** model.
2. Uploaded images are resized, cropped, and normalized.
3. The model predicts the most likely condition from 5 common dog skin diseases:
   - Allergic Dermatitis
   - Hot Spot
   - Fungal Infection
   - Demodectic Mange
   - Pyoderma
4. Users get a clear diagnosis card with medical information.

---

## 🖥️ Tech Stack

- **Python 3**
- **PyTorch** (for the ResNet18 model)
- **Torchvision**
- **Streamlit** (for the interactive UI)
- **PIL** (image processing)

---

## 📂 Project Structure

