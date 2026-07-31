# utils.py
from ultralytics import YOLO
import cv2
import re
import easyocr
import numpy as np
import os

# Initialize EasyOCR reader (this should be done once for efficiency)
reader = easyocr.Reader(['ar'], gpu=False)

# Levenshtein distance for spelling corrections
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

COMMON_ARABIC_NAMES = {
    # Male names
    "احمد", "أحمد", "محمد", "محمود", "مصطفى", "على", "علي", "حسن", "حسين", "خالد", "طارق", "عمر", "عمرو", "ابراهيم", "إبراهيم", 
    "اسماعيل", "إسماعيل", "يوسف", "يحيى", "حمزة", "بلال", "معاذ", "أيمن", "وليد", "كريم", "شريف", "هشام", "هاني", "هانى", 
    "ياسر", "تامر", "رامي", "رامى", "شادي", "شادى", "فادي", "فادى", "عماد", "عامر", "عادل", "كمال", "جمال", "صلاح", 
    "سعيد", "سعد", "مسعد", "رمضان", "شعبان", "رجب", "علاء", "بهاء", "ضياء", "عصام", "مدحت", "رأفت", "رفعت", "ثروت", 
    "حازم", "حاتم", "نادر", "ناجي", "ناجى", "باسم", "باسل", "ماهر", "عاصم", "فؤاد", "فريد", "fawzy", "فوزي", "فوزى", 
    "فتحي", "فتحى", "فرج", "طه", "يحيي", "زكريا", "أشرف", "اشرف", "أمجد", "امجد", "أكرم", "اكرم", "أنور", "انور", 
    "إيهاب", "ايهاب", "وائل", "سامر", "إسلام", "اسلام", "أمير", "امير", "زياد", "عبدالرحمن", "عبدالرحيم", "عبدالعزيز", 
    "عبدالحميد", "عبدالمجيد", "عبدالقادر", "عبداللطيف", "عبدالحليم", "عبدالسلام", "عبدالوهاب", "عبدالفتاح", "عبدالله", 
    "عبدالقوى", "عبدالهادي", "سيد", "صبري", "صبرى", "شوقي", "شوقى", "لطفي", "لطفى", "فهمي", "فهمى", "حلمي", "حلمى", 
    "رمزي", "رمزى", "نجيب", "منير", "سمير", "نبيل", "جميل", "جلال", "كرم", "مراد", "ماجد", "وجدي", "وجدى", "وحيد", 
    "ظافر", "شفيق", "رفيق", "صبحي", "صبخى", "طاهر", "طلعت", "عاطف", "عقيل", "عمران", "عوض", "عيسى", "غالي", "غالى", 
    "غريب", "faiy", "فايز", "فاروق", "فضل", "فيصل", "قاسم", "قطب", "كامل", "metwally", "متولي", "متولى", "محسن", "محفوظ", "مختار", "مروان", 
    "مظهر", "معتز", "معوض", "منصور", "مهدي", "مهدى", "ناصف", "نصار", "نصر", "نعمان", "نعيم", "نهاد", "نور", "هادي", 
    # Female names
    "فاطمة", "فاطمه", "عائشة", "عائشه", "خديجة", "خديجه", "زينب", "رقية", "رقيه", "مريم", "سارة", "sara", "ساره", "منى", "منة", "منه",
    "اميرة", "اميره", "أميرة", "أميره", "ياسمين", "اية", "آية", "اسراء", "إسراء", "دعاء", "شيماء", "نهى", "نهي", "ندى", "ندي",
    "رحمة", "رحمه", "هالة", "هاله", "مي", "مى", "عبير", "رشا", "دينا", "سها", "مها", "ولاء", "هبة", "هبه", "هدى", "هدي"
}

def autocorrect_arabic_name(name):
    if not name:
        return ""
    words = name.split()
    corrected_words = []
    for word in words:
        if word in {'عبد', 'ابو', 'أبو', 'آل', 'الدين'}:
            corrected_words.append(word)
            continue
        if word in COMMON_ARABIC_NAMES:
            corrected_words.append(word)
        else:
            best_match = word
            min_dist = 2
            for common_name in COMMON_ARABIC_NAMES:
                dist = levenshtein_distance(word, common_name)
                if dist < min_dist:
                    min_dist = dist
                    best_match = common_name
            corrected_words.append(best_match)
    return " ".join(corrected_words).strip()

def sauvola_threshold_fast(gray_img, window_size, k=0.15, R=128):
    if window_size % 2 == 0:
        window_size += 1
    gray = gray_img.astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (window_size, window_size))
    sq_mean = cv2.boxFilter(gray * gray, -1, (window_size, window_size))
    variance = sq_mean - (mean * mean)
    variance = np.maximum(variance, 0)
    std_dev = np.sqrt(variance)
    threshold = mean * (1.0 + k * (std_dev / R - 1.0))
    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return binary

def dynamic_ocr_preprocess(bgr_image):
    if bgr_image is None or bgr_image.size == 0:
        return bgr_image
    h, w = bgr_image.shape[:2]
    
    b, g, r = cv2.split(bgr_image)
    b_var = cv2.Laplacian(b, cv2.CV_64F).var()
    g_var = cv2.Laplacian(g, cv2.CV_64F).var()
    r_var = cv2.Laplacian(r, cv2.CV_64F).var()
    
    best_channel = b
    if g_var > b_var and g_var > r_var:
        best_channel = g
    elif r_var > b_var and r_var > g_var:
        best_channel = r
        
    target_h = 96
    scale_factor = 1.0
    if h < target_h:
        scale_factor = target_h / h
        
    if scale_factor > 1.0:
        enhanced = cv2.resize(best_channel, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        h, w = enhanced.shape[:2]
    else:
        enhanced = best_channel.copy()

    sigma_color = int(w * 0.1)
    sigma_space = int(w * 0.1)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    window_size = int(w / 15)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(9, window_size)
    binary = sauvola_threshold_fast(denoised, window_size=window_size, k=0.15)

    blended = np.where(binary == 0, denoised, 255).astype(np.uint8)
    blended = cv2.GaussianBlur(blended, (3, 3), 0)

    padding = 25
    padded = cv2.copyMakeBorder(
        blended, 
        top=padding, 
        bottom=padding, 
        left=padding, 
        right=padding, 
        borderType=cv2.BORDER_CONSTANT, 
        value=255
    )
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

def sort_arabic_ocr_results(results):
    if not results:
        return []
    formatted = []
    for bbox, text, conf in results:
        x0 = min(pt[0] for pt in bbox)
        y0 = min(pt[1] for pt in bbox)
        x1 = max(pt[0] for pt in bbox)
        y1 = max(pt[1] for pt in bbox)
        h = y1 - y0
        formatted.append({
            "bbox": (x0, y0, x1, y1),
            "text": text,
            "y_center": (y0 + y1) / 2.0,
            "x_center": (x0 + x1) / 2.0,
            "height": h
        })
    formatted.sort(key=lambda item: item["y_center"])
    lines = []
    current_line = []
    for item in formatted:
        if not current_line:
            current_line.append(item)
        else:
            avg_height = sum(x["height"] for x in current_line) / len(current_line)
            y_diff = abs(item["y_center"] - current_line[-1]["y_center"])
            if y_diff < (avg_height * 0.7):
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
    if current_line:
        lines.append(current_line)
    sorted_texts = []
    for line in lines:
        line.sort(key=lambda item: item["x_center"], reverse=True)
        sorted_texts.append(" ".join(item["text"] for item in line))
    return sorted_texts

def preprocess_image(cropped_image):
    return dynamic_ocr_preprocess(cropped_image)

def extract_text(image, bbox, field_name):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    if cropped_image.size == 0:
        return ""
        
    # Save raw crop for Streamlit visualization
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(f'output/{field_name}_raw.jpg', cropped_image)
    
    preprocessed_image = preprocess_image(cropped_image)
    
    # Save preprocessed/binarized crop for Streamlit visualization
    cv2.imwrite(f'output/{field_name}_processed.jpg', preprocessed_image)
    
    results = reader.readtext(
        preprocessed_image, 
        detail=1, 
        paragraph=False,
        decoder='beamsearch',
        beamWidth=5,
        contrast_ths=0.1,
        adjust_contrast=False
    )
    sorted_lines = sort_arabic_ocr_results(results)
    text = ' - '.join(sorted_lines).strip()
    return text

def detect_national_id(cropped_image):
    model = YOLO('detect_id.pt')  # Load the model directly in the function
    best_id = ""
    closest_diff = 999
    
    for conf_val in [0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.05]:
        results = model(cropped_image, conf=conf_val)
        detected_info = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                x1 = int(box.xyxy[0][0].item())
                detected_info.append((cls, x1))
                
        detected_info.sort(key=lambda x: x[1])
        id_number = ''.join([str(cls) for cls, _ in detected_info])
        id_number = ''.join(c for c in id_number if c.isdigit())
        
        if len(id_number) == 14:
            return id_number
            
        diff = abs(len(id_number) - 14)
        if diff < closest_diff:
            closest_diff = diff
            best_id = id_number
            
    return best_id

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def expand_bbox_height(bbox, scale=1.2, image_shape=None):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_height = int(height * scale)
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0])
    return [x1, new_y1, x2, new_y2]

def decode_egyptian_id(id_number):
    governorates = {
        '01': 'Cairo', '02': 'Alexandria', '03': 'Port Said', '04': 'Suez',
        '11': 'Damietta', '12': 'Dakahlia', '13': 'Ash Sharqia', '14': 'Kaliobeya',
        '15': 'Kafr El - Sheikh', '16': 'Gharbia', '17': 'Monoufia', '18': 'El Beheira',
        '19': 'Ismailia', '21': 'Giza', '22': 'Beni Suef', '23': 'Fayoum',
        '24': 'El Menia', '25': 'Assiut', '26': 'Sohag', '27': 'Qena',
        '28': 'Aswan', '29': 'Luxor', '31': 'Red Sea', '32': 'New Valley',
        '33': 'Matrouh', '34': 'North Sinai', '35': 'South Sinai', '88': 'Foreign'
    }
    
    # Normalize ID digits and characters
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"
    translation_table = str.maketrans(arabic_digits, english_digits)
    id_number = id_number.translate(translation_table).strip()
    typos = {'o': '0', 'O': '0', 'i': '1', 'I': '1', 'l': '1', '|': '1', '/': '1'}
    id_number = "".join(typos.get(c.lower(), c) for c in id_number if c.isdigit() or c.lower() in typos)
    
    if len(id_number) != 14 or not id_number.isdigit():
        return {
            'Birth Date': 'Unknown',
            'Governorate': 'Unknown',
            'Gender': 'Unknown'
        }
        
    try:
        century_digit = int(id_number[0])
        year = int(id_number[1:3])
        month = int(id_number[3:5])
        day = int(id_number[5:7])
        governorate_code = id_number[7:9]
        gender_code = int(id_number[12:13])

        if century_digit == 2:
            full_year = 1900 + year
        elif century_digit == 3:
            full_year = 2000 + year
        else:
            full_year = 1900 + year

        gender = "Male" if gender_code % 2 != 0 else "Female"
        governorate = governorates.get(governorate_code, "Unknown")
        birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"
    except Exception:
        return {
            'Birth Date': 'Unknown',
            'Governorate': 'Unknown',
            'Gender': 'Unknown'
        }

    return {
        'Birth Date': birth_date,
        'Governorate': governorate,
        'Gender': gender
    }

def process_image(cropped_image):
    model = YOLO('detect_odjects.pt')
    results = model(cropped_image)

    first_name = ''
    second_name = ''
    merged_name = ''
    nid = ''
    address = ''
    serial = ''

    for result in results:
        output_path = 'd2.jpg'
        result.save(output_path)

        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            bbox = [int(coord) for coord in bbox]

            if class_name == 'firstName':
                first_name = extract_text(cropped_image, bbox, 'firstName')
                first_name = autocorrect_arabic_name(first_name)
            elif class_name == 'lastName':
                second_name = extract_text(cropped_image, bbox, 'lastName')
                second_name = autocorrect_arabic_name(second_name)
            elif class_name == 'serial':
                serial = extract_text(cropped_image, bbox, 'serial')
            elif class_name == 'address':
                address = extract_text(cropped_image, bbox, 'address')
            elif class_name == 'nid':
                expanded_bbox = expand_bbox_height(bbox, scale=1.5, image_shape=cropped_image.shape)
                cropped_nid = cropped_image[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                # Save raw NID crop for Streamlit visualization
                os.makedirs('output', exist_ok=True)
                cv2.imwrite('output/nid_raw.jpg', cropped_nid)
                nid = detect_national_id(cropped_nid)

    merged_name = f"{first_name} {second_name}".strip()
    decoded_info = decode_egyptian_id(nid)
    return (first_name, second_name, merged_name, nid, address, decoded_info["Birth Date"], decoded_info["Governorate"], decoded_info["Gender"])

def detect_and_process_id_card(image_path):
    id_card_model = YOLO('detect_id_card.pt')
    id_card_results = id_card_model(image_path)
    image = cv2.imread(image_path)
    
    cropped_image = image
    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image[y1:y2, x1:x2]
            break
            
    return process_image(cropped_image)
