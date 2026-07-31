# generate_synthetic_data.py
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# List of Arabic fonts in the local directory
FONTS = [
    "fonts/NotoNaskhArabic-Regular.ttf",
    "fonts/NotoNaskhArabic-Bold.ttf",
    "fonts/NotoKufiArabic-Regular.ttf",
    "fonts/NotoSansArabic-Regular.ttf"
]

# Lexicon of common names/words to generate
WORDS = [
    "أحمد", "محمد", "محمود", "مصطفى", "علي", "حسن", "حسين", "خالد", "طارق", "عمر", 
    "عبدالرحمن", "عبدالعزيز", "جاسر", "ياسر", "سمير", "تامر", "شريف", "هشام", "هاني",
    "عاصم", "بلال", "معاذ", "أيمن", "وليد", "كريم", "امين", "السيد", "امينه", "وليد",
    "سعد", "مسعد", "صلاح", "جمال", "كمال", "مدحت", "شوقي", "لطفي", "فتحي", "طه",
    "فاطمة", "عائشة", "خديجة", "زينب", "رقية", "مريم", "سارة", "منى", "منة", "ندى",
    "القاهرة", "الجيزة", "الإسكندرية", "بورسعيد", "السويس", "الشرقية", "القليوبية",
    "شارع", "شقة", "الدور", "مصر", "سكن", "داون", "تاون", "أكتوبر", "اول", "رقم"
]

def create_background_pattern(w, h):
    """
    Creates a dynamic background pattern simulating the Guilloche/watermark patterns on ID cards.
    """
    # Create base white/light gray background
    bg = np.ones((h, w, 3), dtype=np.uint8) * random.randint(235, 250)
    
    # 1. Add random faint colorful sine waves/lines
    for _ in range(random.randint(4, 8)):
        color = (random.randint(180, 220), random.randint(180, 220), random.randint(190, 230))
        pts = []
        freq = random.uniform(0.01, 0.05)
        amp = random.uniform(2, 8)
        phase = random.uniform(0, 2 * np.pi)
        start_y = random.randint(0, h)
        for x in range(0, w, 2):
            y = int(start_y + amp * np.sin(freq * x + phase))
            y = max(0, min(h - 1, y))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(bg, pts[i], pts[i+1], color, thickness=1, lineType=cv2.LINE_AA)
            
    # 2. Add random salt-and-pepper background noise
    noise = np.random.randint(-10, 10, (h, w, 3))
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg

def generate_synthetic_image(text, output_path):
    """
    Renders Arabic text using reshaper & bidi, overlays over custom background pattern,
    applies slight rotation, bilateral blur and output formatting.
    """
    w, h = 320, 96
    bg_np = create_background_pattern(w, h)
    
    # Convert to PIL Image to draw text
    img = Image.fromarray(bg_np)
    draw = ImageDraw.Draw(img)
    
    # Select random font
    font_path = random.choice(FONTS)
    font_size = random.randint(28, 36)
    font = ImageFont.truetype(font_path, font_size)
    
    # Reshape and reverse Arabic text
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    
    # Calculate text bounding box to center it
    text_w = draw.textlength(bidi_text, font=font)
    
    # Center text with minor random offset
    x = max(10, (w - text_w) // 2 + random.randint(-15, 15))
    y = max(10, (h - font_size) // 2 + random.randint(-5, 5))
    
    # Render text in near-black color (simulating ink)
    ink_color = (random.randint(15, 40), random.randint(15, 40), random.randint(20, 50))
    draw.text((x, y), bidi_text, font=font, fill=ink_color)
    
    # Convert back to cv2 to apply geometric/optical augmentations
    img_cv = np.array(img)
    
    # 1. Random light rotation (-3 to 3 deg)
    angle = random.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img_cv = cv2.warpAffine(img_cv, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # 2. Apply slight blur to simulate camera focus / anti-aliasing
    if random.random() > 0.5:
        img_cv = cv2.GaussianBlur(img_cv, (3, 3), 0)
        
    cv2.imwrite(output_path, img_cv)

def main():
    os.makedirs("dataset/images", exist_ok=True)
    labels_file = "dataset/labels.txt"
    
    print("Generating synthetic Arabic text dataset...")
    count = 0
    with open(labels_file, "w", encoding="utf-8") as f:
        # Generate single-word names and combinations
        for idx, word in enumerate(WORDS):
            # Single words
            img_name = f"img_{count}.png"
            img_path = os.path.join("dataset/images", img_name)
            generate_synthetic_image(word, img_path)
            f.write(f"{img_name}\t{word}\n")
            count += 1
            
            # Name pairs (e.g. "أحمد محمد")
            for _ in range(3):
                word2 = random.choice(WORDS)
                if word != word2:
                    phrase = f"{word} {word2}"
                    img_name = f"img_{count}.png"
                    img_path = os.path.join("dataset/images", img_name)
                    generate_synthetic_image(phrase, img_path)
                    f.write(f"{img_name}\t{phrase}\n")
                    count += 1
                    
    print(f"Dataset generated successfully! Created {count} images in dataset/images/ and labels.txt.")

if __name__ == "__main__":
    main()
