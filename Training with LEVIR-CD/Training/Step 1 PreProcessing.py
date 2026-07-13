'Preprocessing Script'

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

class LEVIRCDPreprocessor:
    """
    Complete preprocessing pipeline for LEVIR-CD dataset
    - Loads raw images
    - Resizes to target size
    - Normalizes pixel values
    - Saves processed data
    - Generates statistics
    """

 

    
    def __init__(self, raw_data_path, processed_data_path, target_size=(256, 256)):
        """
        Initialize preprocessor
        
        Args:
            raw_data_path: Path to raw LEVIR-CD dataset
            processed_data_path: Path to save processed data
            target_size: Target image size (height, width)
        """

        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.target_size = target_size
        
        # Statistics
        self.stats = {
            'train': {'count': 0, 'changed_pixels': 0, 'total_pixels': 0},
            'val': {'count': 0, 'changed_pixels': 0, 'total_pixels': 0},
            'test': {'count': 0, 'changed_pixels': 0, 'total_pixels': 0}
        }
        
        print("="*70)
        print("LEVIR-CD DATASET PREPROCESSOR")
        print("="*70)
        print(f"Raw data path: {self.raw_data_path}")
        print(f"Processed data path: {self.processed_data_path}")
        print(f"Target size: {self.target_size}")
        print("="*70)
    
    def create_directories(self):
        """Create output directory structure"""
        print("\n[1] Creating directory structure...")
        
        for split in ['train', 'val', 'test']:
            for folder in ['A', 'B', 'label']:
                output_dir = self.processed_data_path / split / folder
                output_dir.mkdir(parents=True, exist_ok=True)
        
        print("✓ Directories created successfully!")
    
    def load_image(self, path, grayscale=False):
        """
        Load image from path
        
        Args:
            path: Image file path
            grayscale: Load as grayscale if True
            
        Returns:
            Loaded image as numpy array
        """
        if grayscale:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def preprocess_image(self, img):
        """
        Preprocess RGB image
        - Resize to target size
        - Normalize to [0, 1]
        
        Args:
            img: Input image (RGB)
            
        Returns:
            Preprocessed image
        """
        # Resize
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Keep as uint8 [0, 255] for storage efficiency
        # Normalization will be done during training
        return img_resized
    
    def preprocess_mask(self, mask):
        """
        Preprocess binary mask
        - Resize to target size
        - Binarize (0 or 255)
        
        Args:
            mask: Input mask (grayscale)
            
        Returns:
            Preprocessed binary mask
        """
        # Resize using nearest neighbor to preserve binary values
        mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Binarize: 0 for no change, 255 for change
        mask_binary = np.where(mask_resized > 127, 255, 0).astype(np.uint8)
        
        return mask_binary
    
    def process_split(self, split_name):
        """
        Process all images in a dataset split
        
        Args:
            split_name: 'train', 'val', or 'test'
        """
        print(f"\n[Processing {split_name.upper()} split]")
        
        # Paths
        split_input_path = self.raw_data_path / split_name
        split_output_path = self.processed_data_path / split_name
        
        # Check if input directory exists
        if not split_input_path.exists():
            print(f"⚠ Warning: {split_input_path} does not exist. Skipping...")
            return
        
        # Get all image files
        img_files = sorted(os.listdir(split_input_path / 'A'))
        
        if len(img_files) == 0:
            print(f"⚠ Warning: No images found in {split_input_path / 'A'}. Skipping...")
            return
        
        print(f"Found {len(img_files)} image pairs")
        
        # Process each image
        for img_file in tqdm(img_files, desc=f"Processing {split_name}"):
            try:
                # Load images
                img_a_path = split_input_path / 'A' / img_file
                img_b_path = split_input_path / 'B' / img_file
                mask_path = split_input_path / 'label' / img_file
                
                img_a = self.load_image(img_a_path, grayscale=False)
                img_b = self.load_image(img_b_path, grayscale=False)
                mask = self.load_image(mask_path, grayscale=True)
                
                # Preprocess
                img_a_proc = self.preprocess_image(img_a)
                img_b_proc = self.preprocess_image(img_b)
                mask_proc = self.preprocess_mask(mask)
                
                # Save processed images
                cv2.imwrite(
                    str(split_output_path / 'A' / img_file),
                    cv2.cvtColor(img_a_proc, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    str(split_output_path / 'B' / img_file),
                    cv2.cvtColor(img_b_proc, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    str(split_output_path / 'label' / img_file),
                    mask_proc
                )
                
                # Update statistics
                self.stats[split_name]['count'] += 1
                self.stats[split_name]['changed_pixels'] += np.sum(mask_proc > 0)
                self.stats[split_name]['total_pixels'] += mask_proc.size
                
            except Exception as e:
                print(f"\n⚠ Error processing {img_file}: {str(e)}")
                continue
        
        print(f"✓ {split_name} split processed: {self.stats[split_name]['count']} images")
    
    def calculate_statistics(self):
        """Calculate and display dataset statistics"""
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        
        total_images = 0
        total_changed = 0
        total_pixels = 0
        
        for split in ['train', 'val', 'test']:
            count = self.stats[split]['count']
            changed = self.stats[split]['changed_pixels']
            total = self.stats[split]['total_pixels']
            
            if total > 0:
                change_ratio = (changed / total) * 100
                print(f"\n{split.upper()} Split:")
                print(f"  Images: {count}")
                print(f"  Changed pixels: {changed:,}")
                print(f"  Total pixels: {total:,}")
                print(f"  Change ratio: {change_ratio:.2f}%")
                
                total_images += count
                total_changed += changed
                total_pixels += total
        
        if total_pixels > 0:
            overall_change_ratio = (total_changed / total_pixels) * 100
            print(f"\nOVERALL:")
            print(f"  Total images: {total_images}")
            print(f"  Total changed pixels: {total_changed:,}")
            print(f"  Total pixels: {total_pixels:,}")
            print(f"  Overall change ratio: {overall_change_ratio:.2f}%")
        
        print("="*70)
    
    def save_statistics(self):
        """Save statistics to JSON file"""
        stats_file = self.processed_data_path / 'preprocessing_stats.json'
        
        stats_output = {
            'preprocessing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'target_size': self.target_size,
            'splits': self.stats
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_output, f, indent=4)
        
        print(f"\n✓ Statistics saved to: {stats_file}")
    
    def visualize_samples(self, split='train', num_samples=3):
        """
        Visualize sample preprocessed images
        
        Args:
            split: Which split to visualize ('train', 'val', 'test')
            num_samples: Number of samples to visualize
        """
        print(f"\n[Visualizing {num_samples} samples from {split} split]")
        
        split_path = self.processed_data_path / split
        img_files = sorted(os.listdir(split_path / 'A'))[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'LEVIR-CD Preprocessed Samples - {split.upper()} Split', 
                     fontsize=16, fontweight='bold')
        
        for i, img_file in enumerate(img_files):
            # Load images
            img_a = self.load_image(split_path / 'A' / img_file)
            img_b = self.load_image(split_path / 'B' / img_file)
            mask = self.load_image(split_path / 'label' / img_file, grayscale=True)
            
            # Plot Before image
            axes[i, 0].imshow(img_a)
            axes[i, 0].set_title('Before (A)', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Plot After image
            axes[i, 1].imshow(img_b)
            axes[i, 1].set_title('After (B)', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            
            # Plot Ground Truth mask
            axes[i, 2].imshow(mask, cmap='gray')
            axes[i, 2].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
            
            # Plot Overlay (changes in red)
            overlay = img_b.copy()
            overlay[mask > 127] = [255, 0, 0]  # Red for changes
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Change Overlay', fontsize=12, fontweight='bold')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.processed_data_path / f'visualization_{split}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")
        
        plt.show()
    
    def run(self, visualize=True):
        """
        Run complete preprocessing pipeline
        
        Args:
            visualize: Whether to create visualization of samples
        """
        print("\nStarting preprocessing pipeline...\n")
        
        # Step 1: Create directories
        self.create_directories()
        
        # Step 2: Process each split
        for split in ['train', 'val', 'test']:
            self.process_split(split)
        
        # Step 3: Calculate and display statistics
        self.calculate_statistics()
        
        # Step 4: Save statistics
        self.save_statistics()
        
        # Step 5: Visualize samples
        if visualize and self.stats['train']['count'] > 0:
            self.visualize_samples(split='train', num_samples=3)
        
        print("\n" + "="*70)
        print("✓ PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"\nProcessed data saved to: {self.processed_data_path}")
        print("\nYou can now proceed to model training!")


def main():
    """Main function to run preprocessing"""
    
    # ============================================
    # CONFIGURATION - MODIFY THESE PATHS
    # ============================================
    
    # Path to raw LEVIR-CD dataset (after extraction)
    RAW_DATA_PATH = "./LEVIR CD"
    
    # Path to save processed data
    PROCESSED_DATA_PATH = "./LEVIR CD Processed"
    
    # Target image size (height, width)
    TARGET_SIZE = (256, 256)
    
    # ============================================
    
    # Create preprocessor
    preprocessor = LEVIRCDPreprocessor(
        raw_data_path=RAW_DATA_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        target_size=TARGET_SIZE
    )
    
    # Run preprocessing
    preprocessor.run(visualize=True)




if __name__ == "__main__":
    main()