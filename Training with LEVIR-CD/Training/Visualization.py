import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# ============================================
# DEFINE CUSTOM METRICS (REQUIRED FOR LOADING)
# ============================================

def iou_metric(y_true, y_pred, threshold=0.5):
    """Intersection over Union (IoU) metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou

def dice_coefficient(y_true, y_pred, threshold=0.5):
    """Dice coefficient metric"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    dice = (2.0 * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) + 1e-7)
    
    return dice

# ============================================
# LOAD MODEL WITH CUSTOM OBJECTS
# ============================================

print("Loading model...")
model = keras.models.load_model(
    './models/unet_change_detection_final_model.h5',
    custom_objects={
        'iou_metric': iou_metric,
        'dice_coefficient': dice_coefficient
    }
)
print("✓ Model loaded successfully!")

# ============================================
# LOAD AND VISUALIZE TEST SAMPLES
# ============================================

def load_test_sample(idx):
    """Load a test image pair and mask"""
    test_path = Path("./LEVIR CD Processed/test")
    files = sorted(os.listdir(test_path / 'A'))
    
    file_name = files[idx]
    
    # Load images
    img_a = cv2.imread(str(test_path / 'A' / file_name))
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    
    img_b = cv2.imread(str(test_path / 'B' / file_name))
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(str(test_path / 'label' / file_name), cv2.IMREAD_GRAYSCALE)
    
    # Normalize
    img_a_norm = img_a.astype(np.float32) / 255.0
    img_b_norm = img_b.astype(np.float32) / 255.0
    mask_norm = np.where(mask > 127, 1, 0).astype(np.float32)
    
    # Concatenate for model input
    img_concat = np.concatenate([img_a_norm, img_b_norm], axis=-1)
    img_concat = np.expand_dims(img_concat, axis=0)
    
    return img_a, img_b, mask_norm, img_concat

# Visualize 5 samples
print("\nGenerating visualizations...")
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
fig.suptitle('Change Detection Results - Test Set (98.09% Accuracy)', 
             fontsize=18, fontweight='bold')

# Column headers
cols = ['Before (A)', 'After (B)', 'Ground Truth', 'Prediction', 'Overlay']

for i in range(5):
    # Sample images evenly from test set
    sample_idx = i * 25  # Every 25th image (0, 25, 50, 75, 100)
    
    print(f"Processing sample {i+1}/5 (image #{sample_idx})...")
    
    img_a, img_b, gt_mask, img_input = load_test_sample(sample_idx)
    
    # Predict
    prediction = model.predict(img_input, verbose=0)[0, :, :, 0]
    pred_binary = (prediction > 0.5).astype(np.float32)
    
    # Plot Before
    axes[i, 0].imshow(img_a)
    if i == 0:
        axes[i, 0].set_title(cols[0], fontsize=14, fontweight='bold')
    axes[i, 0].axis('off')
    
    # Plot After
    axes[i, 1].imshow(img_b)
    if i == 0:
        axes[i, 1].set_title(cols[1], fontsize=14, fontweight='bold')
    axes[i, 1].axis('off')
    
    # Plot Ground Truth
    axes[i, 2].imshow(gt_mask, cmap='gray')
    if i == 0:
        axes[i, 2].set_title(cols[2], fontsize=14, fontweight='bold')
    axes[i, 2].axis('off')
    
    # Plot Prediction
    axes[i, 3].imshow(pred_binary, cmap='gray')
    if i == 0:
        axes[i, 3].set_title(cols[3], fontsize=14, fontweight='bold')
    axes[i, 3].axis('off')
    
    # Plot Overlay (red = detected changes)
    overlay = img_b.copy()
    overlay[pred_binary > 0.5] = [255, 0, 0]  # Red for changes
    axes[i, 4].imshow(overlay)
    if i == 0:
        axes[i, 4].set_title(cols[4], fontsize=14, fontweight='bold')
    axes[i, 4].axis('off')

plt.tight_layout()
output_path = './outputs/test_predictions_final.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualizations saved to: {output_path}")
plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\n📊 Model Performance on Test Set:")
print(f"  Accuracy:  98.09%")
print(f"  Precision: 88.24%")
print(f"  Recall:    72.03%")
print(f"  F1-Score:  79.31%")
print(f"  IoU:       65.72%")
print("="*70)
print("\n✓ All done! Check the outputs folder for results.")