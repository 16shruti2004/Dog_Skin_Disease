# dog_skin_clinic.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Pawsitive Care - Canine Dermatology Scanner",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    :root {
        --primary: #4b8df8;
        --secondary: #1e3a8a;
        --light: #f8fafc;
        --dark: #1e293b;
        --accent: #f59e0b;
    }
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f2f6;
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 2rem;
        border-radius: 0 0 10px 10px;
        margin-bottom: 2rem;
    }
    
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    .btn-primary {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    .result-card {
        border-left: 5px solid var(--accent);
        background: rgba(245, 158, 11, 0.1);
    }
    
    .footer {
        background-color: var(--dark);
        color: white;
        padding: 1rem;
        text-align: center;
        border-radius: 10px;
        margin-top: 2rem;
    }
    
    .file-uploader-area {
        border: 2px dashed var(--primary) !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# Disease Information
DISEASE_DATABASE = {
    "Allergic Dermatitis": {
        "description": "Inflammation caused by environmental or food allergies",
        "symptoms": ["Itching", "Redness", "Hives", "Recurrent ear infections"],
        "treatment": ["Antihistamines", "Allergy testing", "Hypoallergenic diet", "Medicated baths"],
        "severity": "Moderate"
    },
    "Hot Spot": {
        "description": "Acute moist dermatitis causing painful lesions",
        "symptoms": ["Oozing sores", "Intense licking", "Red inflamed skin", "Hair loss"],
        "treatment": ["Topical antibiotics", "Anti-inflammatory medication", "Elizabethan collar"],
        "severity": "Mild to Severe"
    },
    "Fungal Infection": {
        "description": "Yeast or ringworm infection of the skin",
        "symptoms": ["Circular lesions", "Flaky skin", "Itching", "Oily coat"],
        "treatment": ["Antifungal shampoos", "Oral antifungals", "Environmental decontamination"],
        "severity": "Mild"
    },
    "Demodectic Mange": {
        "description": "Mite infestation in hair follicles",
        "symptoms": ["Patchy hair loss", "Scaling", "Redness", "Skin infections"],
        "treatment": ["Medicated dips", "Oral ivermectin", "Antibiotics for secondary infections"],
        "severity": "Moderate to Severe"
    },
    "Pyoderma": {
        "description": "Bacterial skin infection",
        "symptoms": ["Pustules", "Crusting", "Foul odor", "Red inflamed skin"],
        "treatment": ["Oral antibiotics", "Topical therapy", "Identify underlying cause"],
        "severity": "Moderate"
    }
}

def load_model():
    """Load the pre-trained model for predictions."""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(DISEASE_DATABASE))
    return model.eval()

model = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_disease(image):
    """Predict the disease based on the uploaded image."""
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0] * 100
        pred_idx = torch.argmax(probs).item()
    return list(DISEASE_DATABASE.keys())[pred_idx], probs.numpy()

def main():
    # Website Header
    st.markdown("""
    <div class="header">
        <h1 style="margin:0;">🐾 Pawsitive Care</h1>
        <p style="margin:0; opacity:0.9;">Canine Skin Health Scanner</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    st.markdown("""
    <div class="card">
        <h3 style="color:var(--secondary); margin-top:0;">How It Works</h3>
        <ol>
            <li>Upload photo(s) of your dog's skin condition</li>
            <li>Our AI will analyze the image(s)</li>
            <li>Get instant results with treatment recommendations</li>
        </ol>
        <p style="color:var(--secondary); font-size:0.9em;"><em>For best results, upload clear photos of the affected area.</em></p>
    </div>
    """, unsafe_allow_html=True)

    # File Upload Section
    uploaded_files = st.file_uploader(
        "📁 Upload photos of your dog's skin condition",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can upload multiple images for better analysis"
    )

    # Results Section
    if uploaded_files:
        st.markdown(f"### 📋 Analysis Report ({datetime.now().strftime('%Y-%m-%d')})")
        
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption=f"Photo {i+1}", width=300)
                
            with col2:
                prediction, probs = predict_disease(image)
                disease_info = DISEASE_DATABASE[prediction]
                
                st.markdown(f"""
                <div class="card result-card">
                    <h4 style="margin-top:0; color:var(--secondary);">Diagnosis #{i+1}</h4>
                    <p><strong>Condition:</strong> {prediction}</p>
                    <p><strong>Severity:</strong> {disease_info['severity']}</p>
                    <p><strong>Description:</strong> {disease_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Detailed Medical Information"):
                    st.markdown("**Common Symptoms:**")
                    for symptom in disease_info['symptoms']:
                        st.markdown(f"- {symptom}")
                    
                    st.markdown("\n**Recommended Treatments:**")
                    for treatment in disease_info['treatment']:
                        st.markdown(f"- {treatment}")
                
                st.markdown("**Confidence Levels**")
                for disease, prob in zip(DISEASE_DATABASE.keys(), probs):
                    st.progress(int(prob), text=f"{disease}: {prob:.1f}%")

        # Next steps and disclaimer
        st.markdown("""
        <div class="card" style="border-left:5px solid #4b8df8;">
            <h4 style="margin-top:0;">Recommended Actions</h4>
            <p>• Monitor your dog's condition daily</p>
            <p>• Consult with a veterinarian for professional diagnosis</p>
            <p>• Follow veterinary-recommended treatment plans</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="footer">
            <p style="margin:0; font-size:0.9em;">⚠️ Disclaimer: This AI-powered tool provides preliminary information only. 
            It is not a substitute for professional veterinary care.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
