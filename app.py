import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from PIL import Image
import pandas as pd

# CONFIG
st.set_page_config(page_title="Emotion-Based Smile Classifier", layout="wide")

# MODEL SETUP
device = torch.device("cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("scripts/resnet50_smile_model.pth", map_location=device))
model.eval()

class_names = ["anxiety_depression", "fake_smile", "neutral", "real_smile"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

motivation_data = {
    "anxiety & depression": {
        "message": "You're not alone. You're stronger than you think 💙",
        "tip": "Try deep breathing, journaling, or talk to a loved one.",
        "video": "https://www.youtube.com/watch?v=4pLUleLdwY4"
    },
    "fake smile": {
        "message": "It's okay to hide pain. But don’t carry it alone 🤍",
        "tip": "Use humor, write down your thoughts, listen to soft music.",
        "video": "https://www.youtube.com/watch?v=ZToicYcHIOU"
    },
    "neutral": {
        "message": "You're calm. Let’s add a little spark today ✨",
        "tip": "Take a short walk, stretch, or call a friend.",
        "video": "https://www.youtube.com/watch?v=YFSc7Ck0Ao0"
    },
    "real smile": {
        "message": "That’s a beautiful smile! Keep sharing it 😄",
        "tip": "Your joy is powerful. Spread it today!",
        "video": "https://www.youtube.com/watch?v=e0rSmxsVHPE"
    }
}

# PAGE STATE
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_home():
    st.session_state.page = "Home"

def go_classify():
    st.session_state.page = "Classify"

def go_about():
    st.session_state.page = "About"

# NAVBAR
st.markdown("""
<style>
.navbar {
    background-color: #1e272e;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}
.navbar button {
    background-color: #34ace0;
    border: none;
    padding: 10px 20px;
    margin: 5px;
    border-radius: 5px;
    color: white;
    font-weight: bold;
    cursor: pointer;
}
.navbar button:hover {
    background-color: #227093;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.button("🏠 Home", on_click=go_home)
with col2:
    st.button("📸 Classify", on_click=go_classify)
with col3:
    st.button("📖 About", on_click=go_about)

# HOME
if st.session_state.page == "Home":
    st.header("🏠 Welcome to Emotion-Based Smile Classifier")
    st.write("""
    This tool uses AI to detect emotions from facial expressions.
    
    It can identify:
    - 😔 Anxiety & Depression
    - 😐 Neutral Emotion
    - 😅 Fake Smile
    - 😁 Real Smile

    Powered by **Deep Learning** (ResNet50) and built with **Streamlit**.
    """)

# CLASSIFY
elif st.session_state.page == "Classify":
    st.header("📸 Emotion-Based Smile Classifier")

    uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_index = torch.argmax(probs).item()
            pred_class = class_names[pred_index]
            confidence = probs[pred_index].item() * 100

        # Two-column layout
        left, right = st.columns([1, 2])

        with left:
            st.image(image, caption="Uploaded Image", width=300)

        with right:
            st.subheader(f"🎯 Prediction: **{pred_class.replace('_', ' ').title()}** ({confidence:.2f}%)")

            # Graph
            st.markdown("### 📊 Prediction Confidence")
            prob_df = pd.DataFrame({
                "Emotion": class_names,
                "Confidence %": [p.item() * 100 for p in probs]
            }).set_index("Emotion")
            st.bar_chart(prob_df)

            # Explanation
            st.markdown("### 🧠 How the Image is Processed")
            st.markdown("""
            1. 🔄 Image resized to **224x224 pixels**  
            2. 📊 Normalization (ImageNet stats)  
            3. 🤖 Passed through **ResNet50**  
            4. 🧮 Outputs **probabilities** for each emotion  
            5. ✅ Highest score = Final prediction  
            """)

            # Motivation
            st.success(motivation_data[pred_class]["message"])
            st.info(f"💡 Tip: {motivation_data[pred_class]['tip']}")
            st.video(motivation_data[pred_class]["video"])

# ABOUT
elif st.session_state.page == "About":
    st.header("📖 About This Project")
    st.write("""
    - **Project Name**: Emotion-Based Smile Classifier  
    - **Developed by**: C7 
    - **Goal**: Help people understand and track emotional states using AI  
    - **Tech Used**: Python, PyTorch, ResNet50, Streamlit  
    - **GitHub**: [https://github.com/your-repo](https://github.com)  
    - **Contact**: your_email@example.com  
    """)
