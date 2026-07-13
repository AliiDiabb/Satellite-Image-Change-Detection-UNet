"""
LEVIR-CD Change Detection Training Script
2D-CNN (U-Net) for Satellite Image Change Detection
Target: 70%+ Accuracy

FINAL CORRECTED VERSION:
- Fixed mask normalization (prevents negative loss)
- Uses existing train/val/test folders (no splitting needed)
- Saves weights after training completion
- Optimized for RTX 2050 4GB

"""

import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

import albumentations as A
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Training configuration"""
    
    # Paths
    DATA_PATH = "./ChangeDetectionDataset/Real/subset"
    MODEL_SAVE_PATH = "./models"
    OUTPUT_PATH = "./outputs"
    LOGS_PATH = "./logs"
    
    # Model parameters
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 10              # ✅ Set to 2 for RTX 2050 4GB
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Training parameters
    USE_AUGMENTATION = True
    
    # Model architecture
    MODEL_NAME = "unet_change_detection"
    INPUT_CHANNELS = 6  # 3 (RGB from A) + 3 (RGB from B)
    OUTPUT_CHANNELS = 1  # Binary change mask
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 5
    
    # Weight saving options
    SAVE_WEIGHTS_ONLY = True
    SAVE_FINAL_WEIGHTS = True
    
    # Create directories
    @staticmethod
    def create_dirs():
        Path(Config.MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
        Path(Config.OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        Path(Config.LOGS_PATH).mkdir(parents=True, exist_ok=True)


# ============================================
# DATA AUGMENTATION - ✅ FIXED
# ============================================

def get_training_augmentation():
    """Data augmentation for training - FIXED to handle masks properly"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # ✅ FIX: Normalize images only, NOT masks
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        )
    ], additional_targets={'image1': 'image', 'mask': 'mask'})


def get_validation_augmentation():
    """Only normalization for validation/test - FIXED"""
    return A.Compose([
        # ✅ FIX: Normalize images only, NOT masks
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        )
    ], additional_targets={'image1': 'image', 'mask': 'mask'})


# ============================================
# DATA GENERATOR - ✅ COMPLETELY FIXED
# ============================================

class LEVIRCDDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for LEVIR-CD dataset
    FIXED VERSION: Proper mask normalization to [0, 1]
    """
    
    def __init__(self, data_dir, split='train', batch_size=2, 
                 target_size=(256, 256), augmentation=None, shuffle=True):
        """Initialize data generator"""
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        # Get all image files from the specified split folder
        split_path = self.data_dir / split / 'A'
        if not split_path.exists():
            raise ValueError(f"Split path does not exist: {split_path}")
        
        self.image_files = sorted(os.listdir(split_path))
        
        # Initial shuffle (Random)
        if self.shuffle:
            np.random.shuffle(self.image_files)
        
        print(f"✓ Initialized {split} generator: {len(self.image_files)} images, "
              f"{len(self)} batches per epoch")
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_files) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get one batch of data - ✅ FIXED VERSION"""
        # Get batch file indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_files))
        batch_files = self.image_files[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_x = []
        batch_y = []
        
        for file_name in batch_files:
            # Load images from the split folder
            img_a = self._load_image(self.split, 'A', file_name)
            img_b = self._load_image(self.split, 'B', file_name)
            mask = self._load_mask(self.split, file_name)
            
            # ✅ CRITICAL FIX: Convert mask to binary [0, 1] BEFORE augmentation
            mask = np.where(mask > 127, 1, 0).astype(np.uint8)
            
            # Apply augmentation if provided
            if self.augmentation is not None:
                augmented = self.augmentation(
                    image=img_a,
                    image1=img_b,
                    mask=mask
                )
                img_a = augmented['image']
                img_b = augmented['image1']
                mask = augmented['mask']
                
                # ✅ ENSURE mask is float [0, 1] after augmentation
                mask = mask.astype(np.float32)
            else:
                # Simple normalization (no augmentation)
                img_a = img_a.astype(np.float32) / 255.0
                img_b = img_b.astype(np.float32) / 255.0
                mask = mask.astype(np.float32)  # Already 0 or 1 from above
            
            # ✅ SAFETY CHECK: Clip mask values to [0, 1]
            mask = np.clip(mask, 0.0, 1.0)
            
            # Concatenate A and B along channel dimension
            img_concat = np.concatenate([img_a, img_b], axis=-1)
            
            # Expand mask dimensions if needed
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
            
            batch_x.append(img_concat)
            batch_y.append(mask)
        
        # Convert to numpy arrays
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)
        
        # ✅ FINAL SAFETY CHECK: Ensure masks are strictly [0, 1]
        batch_y = np.clip(batch_y, 0.0, 1.0)
        
        # Debug output (only for first batch of first epoch)
        if idx == 0 and self.split == 'train':
            print(f"\n🔍 Data Check:")
            print(f"  Input shape: {batch_x.shape}")
            print(f"  Mask shape: {batch_y.shape}")
            print(f"  Mask min: {batch_y.min():.4f}, max: {batch_y.max():.4f}")
            print(f"  Mask unique values: {np.unique(batch_y)[:10]}")  # Show first 10 unique values
        
        return batch_x, batch_y
    
    def _load_image(self, split, subfolder, filename):
        """Load RGB image from split folder"""
        path = self.data_dir / split / subfolder / filename
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_mask(self, split, filename):
        """Load grayscale mask from split folder"""
        path = self.data_dir / split / 'label' / filename
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {path}")
        return mask
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.image_files)


# ============================================
# MODEL ARCHITECTURE - U-NET
# ============================================

def conv_block(inputs, num_filters, kernel_size=3, padding='same', activation='relu'):
    """Convolutional block with BatchNorm"""
    x = layers.Conv2D(num_filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    x = layers.Conv2D(num_filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    return x


def encoder_block(inputs, num_filters):
    """Encoder block: Conv + MaxPool"""
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    """Decoder block: UpConv + Concatenate + Conv"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape=(256, 256, 6), num_classes=1):
    """Build U-Net architecture for change detection"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bottleneck
    b = conv_block(p4, 1024)
    
    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='unet_change_detection')
    
    return model


# ============================================
# CUSTOM METRICS
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
# TRAINING FUNCTIONS
# ============================================

def create_callbacks(model_name):
    """Create training callbacks"""
    
    checkpoint_path = f"{Config.MODEL_SAVE_PATH}/{model_name}_best_weights.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=Config.SAVE_WEIGHTS_ONLY,
        mode='min',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=Config.REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    )
    
    csv_logger = CSVLogger(
        f"{Config.LOGS_PATH}/{model_name}_training.csv",
        append=False
    )
    
    return [checkpoint, early_stop, reduce_lr, csv_logger]


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['binary_accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_binary_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 0].plot(history.history['iou_metric'], label='Train IoU', linewidth=2)
    axes[1, 0].plot(history.history['val_iou_metric'], label='Val IoU', linewidth=2)
    axes[1, 0].set_title('IoU Metric', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[1, 1].plot(history.history['dice_coefficient'], label='Train Dice', linewidth=2)
    axes[1, 1].plot(history.history['val_dice_coefficient'], label='Val Dice', linewidth=2)
    axes[1, 1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to: {save_path}")
    plt.close()


# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_model(model, test_generator, threshold=0.5):
    """Evaluate model on test set"""
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    all_preds = []
    all_labels = []
    
    # Predict on all test batches
    for i in tqdm(range(len(test_generator)), desc="Evaluating"):
        x_batch, y_batch = test_generator[i]
        preds = model.predict(x_batch, verbose=0)
        
        all_preds.append((preds > threshold).astype(np.float32))
        all_labels.append(y_batch)
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Flatten for metrics calculation
    y_true_flat = all_labels.flatten()
    y_pred_flat = all_preds.flatten()
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Calculate IoU manually
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    iou = intersection / (union + 1e-7)
    
    # Print results
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  IoU:       {iou*100:.2f}%")
    print("="*70)
    
    # Save results to JSON
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'iou': float(iou),
        'threshold': threshold,
        'test_samples': len(all_labels)
    }
    
    results_path = f"{Config.OUTPUT_PATH}/test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    return results, all_preds, all_labels


def visualize_predictions(model, test_generator, num_samples=5):
    """Visualize model predictions"""
    print(f"\n[Visualizing {num_samples} predictions]")
    
    # Get a batch
    x_batch, y_batch = test_generator[0]
    predictions = model.predict(x_batch[:num_samples], verbose=0)
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, num_samples*4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Change Detection Results', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # Extract A and B images
        img_a = x_batch[i, :, :, :3]
        img_b = x_batch[i, :, :, 3:6]
        gt_mask = y_batch[i, :, :, 0]
        pred_mask = predictions[i, :, :, 0]
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        
        # Denormalize images for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_a = (img_a * std + mean).clip(0, 1)
        img_b = (img_b * std + mean).clip(0, 1)
        
        # Plot Before
        axes[i, 0].imshow(img_a)
        axes[i, 0].set_title('Before (A)')
        axes[i, 0].axis('off')
        
        # Plot After
        axes[i, 1].imshow(img_b)
        axes[i, 1].set_title('After (B)')
        axes[i, 1].axis('off')
        
        # Plot Ground Truth
        axes[i, 2].imshow(gt_mask, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Plot Prediction
        axes[i, 3].imshow(pred_binary, cmap='gray')
        axes[i, 3].set_title('Prediction')
        axes[i, 3].axis('off')
        
        # Plot Overlay
        overlay = (img_b * 255).astype(np.uint8)
        overlay[pred_binary > 0.5] = [255, 0, 0]  # Red for predicted changes
        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title('Prediction Overlay')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    save_path = f"{Config.OUTPUT_PATH}/predictions_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Predictions saved to: {save_path}")
    plt.close()


# ============================================
# MAIN TRAINING SCRIPT
# ============================================

def main():
    """Main training function"""
    
    print("="*70)
    print("LEVIR-CD CHANGE DETECTION TRAINING")
    print("2D-CNN (U-Net) Architecture - FIXED VERSION")
    print("="*70)
    print(f"Data path: {Config.DATA_PATH}")
    print(f"Image size: {Config.IMAGE_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print("="*70)
    print("✅ Using existing train/val/test splits from preprocessing")
    print("✅ Fixed mask normalization (prevents negative loss)")
    print("✅ Will save model weights after training")
    print("="*70)
    
    # Create directories
    Config.create_dirs()
    
    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\n✓ GPU detected: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("\n⚠ No GPU detected. Training will use CPU (very slow!)")
    
    # Create data generators
    print("\n[1] Creating data generators from existing splits...")
    
    train_aug = get_training_augmentation() if Config.USE_AUGMENTATION else None
    val_aug = get_validation_augmentation()
    
    train_generator = LEVIRCDDataGenerator(
        data_dir=Config.DATA_PATH,
        split='train',
        batch_size=Config.BATCH_SIZE,
        target_size=Config.IMAGE_SIZE,
        augmentation=train_aug,
        shuffle=True
    )
    
    val_generator = LEVIRCDDataGenerator(
        data_dir=Config.DATA_PATH,
        split='val',
        batch_size=Config.BATCH_SIZE,
        target_size=Config.IMAGE_SIZE,
        augmentation=val_aug,
        shuffle=False
    )
    
    test_generator = LEVIRCDDataGenerator(
        data_dir=Config.DATA_PATH,
        split='test',
        batch_size=Config.BATCH_SIZE,
        target_size=Config.IMAGE_SIZE,
        augmentation=val_aug,
        shuffle=False
    )
    
    # Build model
    print("\n[2] Building U-Net model...")
    model = build_unet(
        input_shape=(*Config.IMAGE_SIZE, Config.INPUT_CHANNELS),
        num_classes=Config.OUTPUT_CHANNELS
    )
    
    # Compile model
    print("\n[3] Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(name='binary_accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            iou_metric,
            dice_coefficient
        ]
    )
    
    # Print model summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model.summary()
    print("="*70)
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Create callbacks
    callbacks = create_callbacks(Config.MODEL_NAME)
    
    # Train model
    print("\n[4] Starting training...")
    print("="*70)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    
    # Save final weights after training
    if Config.SAVE_FINAL_WEIGHTS:
        print("\n[5] Saving final model weights...")
        final_weights_path = f"{Config.MODEL_SAVE_PATH}/{Config.MODEL_NAME}_final_weights.h5"
        model.save_weights(final_weights_path)
        print(f"✓ Final weights saved to: {final_weights_path}")
        
        # Also save the complete model
        final_model_path = f"{Config.MODEL_SAVE_PATH}/{Config.MODEL_NAME}_final_model.h5"
        model.save(final_model_path)
        print(f"✓ Final complete model saved to: {final_model_path}")
    
    # Plot training history
    print("\n[6] Plotting training history...")
    plot_training_history(
        history,
        f"{Config.OUTPUT_PATH}/training_history.png"
    )
    
    # Load best model weights
    print("\n[7] Loading best model weights...")
    best_weights_path = f"{Config.MODEL_SAVE_PATH}/{Config.MODEL_NAME}_best_weights.h5"
    
    # Rebuild model and load best weights
    best_model = build_unet(
        input_shape=(*Config.IMAGE_SIZE, Config.INPUT_CHANNELS),
        num_classes=Config.OUTPUT_CHANNELS
    )
    best_model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(name='binary_accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            iou_metric,
            dice_coefficient
        ]
    )
    best_model.load_weights(best_weights_path)
    print(f"✓ Loaded best weights from: {best_weights_path}")
    
    # Evaluate on test set
    print("\n[8] Evaluating on test set...")
    results, predictions, labels = evaluate_model(best_model, test_generator)
    
    # Visualize predictions
    print("\n[9] Visualizing predictions...")
    visualize_predictions(best_model, test_generator, num_samples=5)
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Saved Files:")
    print(f"  Best weights: {best_weights_path}")
    if Config.SAVE_FINAL_WEIGHTS:
        print(f"  Final weights: {final_weights_path}")
        print(f"  Final model: {final_model_path}")
    print(f"\n📊 Test Results:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  IoU: {results['iou']*100:.2f}%")
    print(f"  F1-Score: {results['f1_score']*100:.2f}%")
    print("="*70)
    
    if results['accuracy'] >= 0.70:
        print("\n🎉 SUCCESS! Target accuracy (70%) achieved!")
    else:
        print(f"\n⚠ Accuracy ({results['accuracy']*100:.2f}%) below target (70%). Consider:")
        print("  - Training for more epochs")
        print("  - Adjusting learning rate")
        print("  - Using stronger data augmentation")
    
    print(f"\nAll outputs saved to: {Config.OUTPUT_PATH}")
    print(f"Training logs saved to: {Config.LOGS_PATH}")
    print("\n✓ Done!")


if __name__ == "__main__":
    main()