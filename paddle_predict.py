# paddle_predict.py
import sys
import os
import warnings

# Disable MKLDNN CPU conflict logs/errors in PaddlePaddle
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
os.environ['PP_LOG_LEVEL'] = '3' # Suppress warnings

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    # Suppress stderr to prevent warning spam
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='ar', enable_mkldnn=False)
    
    # Restore stderr
    sys.stderr = stderr
    
    if len(sys.argv) < 2:
        print("")
        sys.exit(0)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("")
        sys.exit(0)
        
    result = ocr.ocr(image_path)
    if result and isinstance(result, list) and len(result) > 0:
        res_dict = result[0]
        if isinstance(res_dict, dict) and 'rec_texts' in res_dict:
            # Reverse the Left-to-Right sorted layout to restore Arabic Right-to-Left reading order
            rec_texts = list(reversed(res_dict['rec_texts']))
            print(" ".join(rec_texts).strip())
            sys.exit(0)
            
    print("")
except Exception as e:
    print("")
