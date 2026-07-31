import os
import tempfile
from PIL import Image
import streamlit as st
from utils import detect_and_process_id_card

# Streamlit configuration
st.set_page_config(page_title='ID Egyptian Card ', page_icon='💳', layout='wide')

# Initialize session state for navigation
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Home"

# Sidebar navigation menu
tabs = ["Home", "Guide"]
selected_tab = st.sidebar.radio("Navigation", tabs)

# Update the session state with the selected tab
st.session_state.current_tab = selected_tab

# Home Tab
if st.session_state.current_tab == "Home":
    uploaded_file = st.sidebar.file_uploader("Upload an ID card image",
                                             type=['webp', 'jpg', 'tif', 'tiff', 'png', 'mpo', 'bmp', 'jpeg', 'dng', 'pfm'])

    # If no file is uploaded, display the HOME image
    if not uploaded_file:
        st.image("ocr2.png", use_container_width=True)
    else:
        # Clean old visual crops from output folder to avoid showing stale data
        import shutil
        if os.path.exists("output"):
            for f in os.listdir("output"):
                if f.endswith(".jpg"):
                    try:
                        os.remove(os.path.join("output", f))
                    except Exception:
                        pass
        
        # If a file is uploaded, process it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        image = Image.open(temp_file_path)

        st.subheader('Egyptian ID Card EXTRACTING, OCR 💳')
        st.sidebar.image(image)

        try:
            # Call the detect_and_process_id_card function
            first_name, second_name, Full_name, national_id, address, birth, gov, gender = detect_and_process_id_card(temp_file_path)
            
            # Show the cropped card image detected by YOLO
            st.image(Image.open("d2.jpg"), caption="Cropped Card Bounding Box", use_container_width=True)
            
            st.markdown("---")
            
            # 🔍 Interactive Preprocessing Pipeline Expander
            with st.expander("🔍 Show Image Preprocessing & Sauvola Phases", expanded=True):
                st.markdown("### Preprocessing Pipeline Visualization")
                st.markdown("Compare the original crops with our adaptive Sauvola binarization + edge-softening filters.")
                
                # Check and display firstName
                if os.path.exists("output/firstName_raw.jpg") and os.path.exists("output/firstName_processed.jpg"):
                    st.markdown("#### 1. First Name Field (`firstName`)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image("output/firstName_raw.jpg", caption="Original Crop", use_container_width=True)
                    with col2:
                        st.image("output/firstName_processed.jpg", caption="Sauvola Preprocessed (Grayscale Blend + Softening)", use_container_width=True)
                
                # Check and display lastName
                if os.path.exists("output/lastName_raw.jpg") and os.path.exists("output/lastName_processed.jpg"):
                    st.markdown("#### 2. Last Name Field (`lastName`)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image("output/lastName_raw.jpg", caption="Original Crop", use_container_width=True)
                    with col2:
                        st.image("output/lastName_processed.jpg", caption="Sauvola Preprocessed (Grayscale Blend + Softening)", use_container_width=True)
                
                # Check and display address
                if os.path.exists("output/address_raw.jpg") and os.path.exists("output/address_processed.jpg"):
                    st.markdown("#### 3. Address Field (`address`)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image("output/address_raw.jpg", caption="Original Crop", use_container_width=True)
                    with col2:
                        st.image("output/address_processed.jpg", caption="Sauvola Preprocessed (Grayscale Blend + Softening)", use_container_width=True)

                # Check and display NID
                if os.path.exists("output/nid_raw.jpg"):
                    st.markdown("#### 4. National ID Number Field (`nid`)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image("output/nid_raw.jpg", caption="Original Crop", use_container_width=True)
                    with col2:
                        st.write("Digits extraction is directly processed using YOLO Digit Sweep (conf 0.25 -> 0.05).")

            st.markdown("---")
            st.markdown(" ## WORDS EXTRACTED : ")
            st.write(f"First Name: {first_name}")
            st.write(f"Second Name: {second_name}")
            st.write(f"Full Name: {Full_name}")
            st.write(f"National ID: {national_id}")
            st.write(f"Address: {address}")
            st.write(f"Birth Date: {birth}")
            st.write(f"Governorate: {gov}")
            st.write(f"Gender: {gender}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            os.remove(temp_file_path)

# Documentation Tab
elif st.session_state.current_tab == "Guide":
    st.title("How to use our application 📖")
    st.write("""
    ## Project Overview:
    This application processes Egyptian ID cards to extract key information, including names, addresses, and national IDs.  
    It also decodes the national ID to provide additional details like birth date, governorate, and gender.

    ## Features:
    - **ID Card Detection**: Automatically detects and crops the ID card from the image.
    - **Field Detection**: Identifies key fields such as first name, last name, address, and serial number.
    - **Text Extraction**: Extracts Arabic and English text using EasyOCR.
    - **National ID Decoding**: Decodes the ID to extract:
        - Birth Date
        - Governorate
        - Gender
        - Birthplace
        - Location
        - Nationality

    ## How It Works:
    1. **Upload an Image**: Upload an image of the ID card using the sidebar.
    2. **Detection and Extraction**:
        - YOLO models detect the ID card and its fields.
        - EasyOCR extracts text from the identified fields.
    3. **Result Presentation**:
        - Outputs extracted information such as full name, address, and national ID details.
    4. **ID Decoding**:
        - Decodes the national ID to reveal demographic details.

    ## Steps to Use:
    - Get your image ready.
    - Click on Home.
    - Upload an Egyptian ID card image.
    - View the extracted information and analysis.
        
    ## HOPE YOU ENJOY THE EXPERIENCE 💖
    """)
