# predict.py
import sys
import torch
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from train_recognition import CRNN, ALPHABET, IDX_TO_CHAR

def greedy_decoder(output):
    """
    Decodes CTC network outputs (greedy search).
    """
    # output shape: [seq_len, batch_size, num_classes] -> we assume batch_size=1
    arg_maxes = torch.argmax(output, dim=2).squeeze(1).tolist()
    
    # Merge consecutive duplicate tokens and remove blank token (index 0)
    decoded_indices = []
    prev_idx = -1
    for idx in arg_maxes:
        if idx != prev_idx:
            if idx != 0: # 0 is standard CTC blank token
                decoded_indices.append(idx)
            prev_idx = idx
            
    # Convert indices back to string characters
    decoded_text = "".join(IDX_TO_CHAR.get(idx, "") for idx in decoded_indices)
    return decoded_text

def predict(image_path, weights_path="custom_arabic_recognition.pth"):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    target_w, target_h = 256, 64
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # Initialize model
    model = CRNN()
    
    # Load weights
    if not torch.cuda.is_available():
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Run prediction
    with torch.no_grad():
        outputs = model(img_tensor) # Shape: [seq_len, 1, num_classes]
        raw_decoded = greedy_decoder(outputs)
        
    # Reshape Arabic text for terminal display/use
    reshaped_text = arabic_reshaper.reshape(raw_decoded)
    bidi_text = get_display(reshaped_text)
    
    return raw_decoded, bidi_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_image_crop> [weights_path]")
        sys.exit(1)
        
    img_path = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else "custom_arabic_recognition.pth"
    
    raw_text, visual_text = predict(img_path, weights)
    print(f"\n--- Prediction Results ---")
    print(f"Logical text output: {raw_text}")
    print(f"Visual text output:  {visual_text}\n")
