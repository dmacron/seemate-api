from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
import logging
import base64

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI model once on startup
logger.info("Loading image captioning model...")
try:
    # Using smaller, faster model for Railway deployment
     pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    pipe = None

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "SeeMate AI Vision API",
        "model_status": "loaded" if pipe else "failed"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze image sent directly as file upload or base64
    
    Method 1 (Recommended): Multipart form-data
    - Send image as file with key 'image'
    
    Method 2: JSON with base64
    - Send JSON: {"image_base64": "..."}
    
    Returns: {"success": true, "caption": "...", "message": "..."}
    """
    try:
        frame = None
        
        # Method 1: File upload (from ESP32 or Flutter)
        if 'image' in request.files:
            file = request.files['image']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            logger.info("📸 Received image via file upload")
        
        # Method 2: Base64 encoded image (alternative)
        elif request.is_json:
            data = request.get_json()
            if 'image_base64' in data:
                img_data = base64.b64decode(data['image_base64'])
                img_array = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                logger.info("📸 Received image via base64")
        
        if frame is None:
            return jsonify({
                "success": False,
                "error": "No valid image received. Send as 'image' file or 'image_base64' in JSON"
            }), 400
        
        # Check if model is loaded
        if pipe is None:
            return jsonify({
                "success": False,
                "error": "AI model not loaded on server"
            }), 503
        
        # Convert image for AI processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        logger.info("🤖 Analyzing image...")
        
        # Generate caption
        result = pipe(pil_image)
        
        if result and len(result) > 0 and 'generated_text' in result[0]:
            caption = result[0]['generated_text']
            logger.info(f"✅ Caption generated: {caption}")
            
            return jsonify({
                "success": True,
                "caption": caption,
                "message": f"I can see: {caption}"
            }), 200
        else:
            logger.warning("⚠️ No caption generated")
            return jsonify({
                "success": False,
                "error": "Could not generate caption"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Processing error: {str(e)}"
        }), 500

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify server is working"""
    return jsonify({
        "message": "SeeMate API is working!",
        "endpoints": {
            "/": "Health check",
            "/analyze": "POST image for analysis (multipart or base64)",
            "/test": "This test endpoint"
        }
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting SeeMate API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)