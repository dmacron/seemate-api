from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import pyttsx3
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI model
logger.info("Loading image captioning model...")
pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
logger.info("Model loaded!")

# Initialize TTS (for server-side, but you'll use Flutter TTS instead)
# tts_engine = pyttsx3.init()

def capture_image_from_url(url, timeout=10, max_retries=3):
    """Capture a single image from ESP32 /capture endpoint"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Capturing image (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                image_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    logger.info("Successfully captured image from ESP32")
                    return frame, None
                else:
                    return None, "Failed to decode image"
            else:
                return None, f"HTTP Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return None, "Connection timeout"
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                return None, "Cannot connect to ESP32"
        except Exception as e:
            return None, str(e)
    
    return None, "Max retries reached"

@app.route('/', methods=['GET'])
def home():
    """Health check"""
    return jsonify({
        "status": "online",
        "service": "SeeMate Vision API"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze image from ESP32
    Expects JSON: {"esp32_url": "http://192.168.0.144/capture"}
    Returns JSON: {"success": true/false, "caption": "...", "error": "..."}
    """
    try:
        data = request.get_json()
        
        if not data or 'esp32_url' not in data:
            return jsonify({
                "success": False,
                "error": "esp32_url required"
            }), 400
        
        esp32_url = data['esp32_url']
        logger.info(f"Analyzing from: {esp32_url}")
        
        # Capture from ESP32
        frame, error = capture_image_from_url(esp32_url)
        if error:
            logger.error(f"Capture failed: {error}")
            return jsonify({"success": False, "error": error}), 400
        
        # Analyze image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        result = pipe(pil_image)
        
        if result and len(result) > 0 and 'generated_text' in result[0]:
            caption = result[0]['generated_text']
            logger.info(f"Caption: {caption}")
            
            return jsonify({
                "success": True,
                "caption": caption,
                "message": f"I can see: {caption}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "No caption generated"
            }), 500
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)