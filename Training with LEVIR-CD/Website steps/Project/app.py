from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import json

# ============================================
# INITIALIZE FLASK APP
# ============================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULT_FOLDER'] = './results'
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 16MB max file size

# Create necessary folders
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['RESULT_FOLDER']).mkdir(parents=True, exist_ok=True)

# ============================================
# LOAD TRAINED MODEL
# ============================================

def iou_metric(y_true, y_pred, threshold=0.5):
    """IoU metric for model loading"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def dice_coefficient(y_true, y_pred, threshold=0.5):
    """Dice coefficient for model loading"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    return (2.0 * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) + 1e-7)

print("Loading trained model...")
model = keras.models.load_model(
    './models/W3.h5',
    custom_objects={
        'iou_metric': iou_metric,
        'dice_coefficient': dice_coefficient
    }
)
print("✓ Model loaded successfully!")

# ============================================
# IMAGE PROCESSING FUNCTIONS
# ============================================

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess uploaded image - MATCHES TRAINING"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0
    
    # ✅ Apply ImageNet normalization (SAME AS TRAINING!)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std
    
    return img_resized, img_normalized

def detect_changes(image_before_path, image_after_path):
    """Detect changes between two images"""
    # Load and preprocess images
    img_before, img_before_norm = preprocess_image(image_before_path)
    img_after, img_after_norm = preprocess_image(image_after_path)
    
    
    print(f"\n🔍 IMAGE COMPARISON:")
    print(f"Before shape: {img_before.shape}")
    print(f"After shape: {img_after.shape}")
    
    # Check if images are identical
    difference = np.abs(img_before.astype(float) - img_after.astype(float)).mean()
    print(f"Average pixel difference: {difference:.2f}")
    
    if difference < 1.0:
        print("⚠️  WARNING: Images are nearly IDENTICAL!")
    else:
        print(f"✅ Images are different (diff: {difference:.2f})")
    
    # Concatenate images
    img_concat = np.concatenate([img_before_norm, img_after_norm], axis=-1)
    img_concat = np.expand_dims(img_concat, axis=0)
    
    # Predict
    prediction = model.predict(img_concat, verbose=0)[0, :, :, 0]
    print(f"\n🔍 DEBUG INFO:")
    print(f"Prediction min: {prediction.min():.4f}")
    print(f"Prediction max: {prediction.max():.4f}")  # This is your "Max Confidence"
    print(f"Prediction mean: {prediction.mean():.4f}")
    print(f"Pixels > 0.5: {np.sum(prediction > 0.5)}")
    print(f"Pixels > 0.3: {np.sum(prediction > 0.3)}")
    print(f"Pixels > 0.1: {np.sum(prediction > 0.1)}\n")
    prediction_binary = (prediction > 0.3).astype(np.uint8) * 255
    
    # Create overlay
    overlay = img_after.copy()
    overlay[prediction_binary > 127] = [255, 0, 0]  # Red for changes
    
    return {
        'before': img_before,
        'after': img_after,
        'prediction': prediction_binary,
        'overlay': overlay,
        'confidence': float(prediction.max())
    }

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string for web display"""
    img_pil = Image.fromarray(img_array.astype('uint8'))
    buffer = BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Process uploaded images and return results"""
    try:
        # Check if files are present
        if 'image_before' not in request.files or 'image_after' not in request.files:
            return jsonify({'error': 'Both images are required'}), 400
        
        file_before = request.files['image_before']
        file_after = request.files['image_after']
        
        if file_before.filename == '' or file_after.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Save uploaded files
        before_path = Path(app.config['UPLOAD_FOLDER']) / 'before.png'
        after_path = Path(app.config['UPLOAD_FOLDER']) / 'after.png'
        
        file_before.save(before_path)
        file_after.save(after_path)
        
        # Detect changes
        results = detect_changes(before_path, after_path)
        
        # Convert images to base64 for display
        response = {
            'success': True,
            'images': {
                'before': numpy_to_base64(results['before']),
                'after': numpy_to_base64(results['after']),
                'prediction': numpy_to_base64(results['prediction']),
                'overlay': numpy_to_base64(results['overlay'])
            },
            'stats': {
                'confidence': f"{results['confidence']*100:.2f}%",
                'changed_pixels': int(np.sum(results['prediction'] > 127)),
                'total_pixels': results['prediction'].size,
                'change_percentage': f"{(np.sum(results['prediction'] > 127) / results['prediction'].size) * 100:.2f}%"
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("CHANGE DETECTION PROTOTYPE - WEB INTERFACE")
    print("="*70)
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)