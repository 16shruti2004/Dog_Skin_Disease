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
📦 Pawsitive-Care/
├── dog_skin_clinic.py # Main Streamlit app
├── README.md # Project README
├── requirements.txt # Python dependencies (optional)


---

## ⚙️ How to Run

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/pawsitive-care.git
   cd pawsitive-care
2. Install dependencies

bash
Copy
Edit
pip install streamlit torch torchvision pillow

3. Run the app

bash
Copy
Edit
streamlit run dog_skin_clinic.py

4. Open in your browser

Streamlit will show a local link (like http://localhost:8501). Click and test it!

## 📝 Disclaimer
This tool is intended for informational purposes only. It is not a substitute for professional veterinary care. Always consult a licensed vet for diagnosis and treatment.

## 💡 Future Improvements
Add more disease classes with larger dataset

Improve model accuracy with real veterinary data

Deploy online for public use (Streamlit Cloud, Heroku, etc.)

Build a mobile app version for easy use by pet owners

## 🙏 Acknowledgements
Built with ❤️ by Shruti Kumari Hela
Guidance: Ms. Arpita Roy — NSTIW Kolkata

## 🐕✨ Helping pet owners stay informed — one wag at a time!


---

**✅ Just copy this into a `README.md` and push it to your GitHub!**  
If you want, I can also write you a short `LICENSE` or `.gitignore` — just say *yes*!




