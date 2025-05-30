#!/usr/bin/env python3
"""
Palm Leaf Disease Classification Training Script
Designed for high accuracy in real-world scenarios with robust preprocessing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PalmDiseaseClassifier:
    def __init__(self, data_dir, img_size=224, batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = []
        
        # Disease classes based on the research paper (excluding leaf spots as specified)
        # 3 physiological + 4 fungal + 1 pest + healthy
        self.disease_classes = [
            'healthy',
            'potassium_deficiency',
            'manganese_deficiency', 
            'magnesium_deficiency',
            'fungal_disease_1',  # Replace with actual fungal disease names from your dataset
            'fungal_disease_2',  # Replace with actual fungal disease names from your dataset
            'fungal_disease_3',  # Replace with actual fungal disease names from your dataset
            'fungal_disease_4',  # Replace with actual fungal disease names from your dataset
            'pest_damage'        # Replace with actual pest disorder name from your dataset
        ]
        
    def advanced_preprocessing(self, image_path):
        """
        Advanced preprocessing pipeline for raw images
        Handles various real-world challenges like lighting, blur, noise
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Noise reduction
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        img = cv2.merge((l_channel, a_channel, b_channel))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        
        # Resize while maintaining aspect ratio
        h, w = img.shape[:2]
        if h > w:
            new_h, new_w = self.img_size, int(w * self.img_size / h)
        else:
            new_h, new_w = int(h * self.img_size / w), self.img_size
            
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Pad to square
        delta_w = self.img_size - new_w
        delta_h = self.img_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def get_augmentation_pipeline(self):
        """
        Comprehensive data augmentation for real-world robustness
        """
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, 
                              rotate_limit=45, p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def standardize_extensions(dataset_path, dry_run=True):
        """
        Utility to standardize image file extensions to lowercase
        
        Args:
            dataset_path (str): Path to dataset directory
            dry_run (bool): If True, only reports what would be changed
        """
        dataset_path = Path(dataset_path)
        changes = []
        
        # Find all image files with uppercase extensions
        uppercase_patterns = ['*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
        
        for pattern in uppercase_patterns:
            for file_path in dataset_path.rglob(pattern):
                new_name = file_path.stem + file_path.suffix.lower()
                new_path = file_path.parent / new_name
                
                if new_path != file_path:
                    changes.append((file_path, new_path))
        
        if dry_run:
            print(f"Found {len(changes)} files with uppercase extensions:")
            for old_path, new_path in changes[:10]:  # Show first 10
                print(f"  {old_path.name} -> {new_path.name}")
            if len(changes) > 10:
                print(f"  ... and {len(changes) - 10} more")
            print(f"\nRun with dry_run=False to actually rename files")
        else:
            print(f"Renaming {len(changes)} files...")
            for old_path, new_path in changes:
                try:
                    old_path.rename(new_path)
                    print(f"Renamed: {old_path.name} -> {new_path.name}")
                except Exception as e:
                    print(f"Error renaming {old_path}: {e}")
        
        return len(changes)
        """
        Validate dataset structure and report any issues
        """
        print("Validating dataset structure...")
        
        total_images = 0
        issues_found = []
        
        for class_name in self.disease_classes:
            class_path = self.data_dir / class_name
            
            if not class_path.exists():
                issues_found.append(f"Missing class directory: {class_name}")
                continue
            
            # Find all image files (case-insensitive)
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                              '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
            
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(class_path.glob(ext)))
            
            # Remove duplicates
            image_files = list(set(image_files))
            
            # Check for valid images
            valid_images = 0
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        valid_images += 1
                    else:
                        issues_found.append(f"Corrupted image: {img_path}")
                except Exception as e:
                    issues_found.append(f"Error reading {img_path}: {str(e)}")
            
            print(f"{class_name}: {valid_images} valid images")
            total_images += valid_images
            
            if valid_images < 50:
                issues_found.append(f"Low image count for {class_name}: {valid_images} (recommend >100)")
        
        print(f"\nTotal valid images: {total_images}")
        
        if issues_found:
            print(f"\nIssues found ({len(issues_found)}):")
            for issue in issues_found:
                print(f"  - {issue}")
        else:
            print("\nNo issues found! Dataset structure looks good.")
        
        return len(issues_found) == 0
        """
        Load and prepare dataset with advanced preprocessing
        """
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.disease_classes):
            class_path = self.data_dir / class_name
            if not class_path.exists():
                print(f"Warning: Class directory {class_name} not found")
                continue
            
            # Robust image file detection - handles both uppercase and lowercase extensions
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                              '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
            
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(class_path.glob(ext)))
            
            # Remove duplicates (in case of case-insensitive file systems)
            image_files = list(set(image_files))
            
            print(f"Processing {len(image_files)} images for class: {class_name}")
            
            for img_path in image_files:
                processed_img = self.advanced_preprocessing(img_path)
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(class_idx)
                else:
                    print(f"Warning: Could not process image {img_path}")
        
        self.class_names = self.disease_classes
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Total images loaded: {len(X)}")
        print(f"Image shape: {X[0].shape}")
        print(f"Classes: {self.class_names}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, len(self.class_names))
        y_val = keras.utils.to_categorical(y_val, len(self.class_names))
        y_test = keras.utils.to_categorical(y_test, len(self.class_names))
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_model(self, num_classes):
        """
        Create an advanced CNN model with transfer learning
        """
        # Base model with pre-trained weights
        base_model = applications.EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Fine-tune the last few layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 20
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Custom top layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        return model
    
    def train_model(self, train_data, val_data, epochs=100):
        """
        Train the model with advanced techniques
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create model
        self.model = self.create_model(len(self.class_names))
        
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(learning_rate=0.0001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        # Calculate class weights for imbalanced dataset
        y_train_classes = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train_classes), 
            y=y_train_classes
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_palm_disease_model.keras',  # Using .keras format (recommended)
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training...")
        print(f"Model architecture:")
        self.model.summary()
        
        # Training
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, test_data):
        """
        Comprehensive model evaluation
        """
        X_test, y_test = test_data
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_top2_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-2 Accuracy: {test_top2_acc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_acc, test_top2_acc
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-2 Accuracy
        axes[1, 0].plot(self.history.history['top_2_accuracy'], label='Train Top-2 Acc')
        axes[1, 0].plot(self.history.history['val_top_2_accuracy'], label='Val Top-2 Acc')
        axes[1, 0].set_title('Top-2 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-2 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """
        Predict disease for a single image
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        # Preprocess image
        img = self.advanced_preprocessing(image_path)
        if img is None:
            print("Could not process image")
            return None
            
        # Predict
        img_batch = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img_batch)
        
        # Get top predictions
        pred_probs = predictions[0]
        top_indices = np.argsort(pred_probs)[::-1][:3]
        
        print(f"Predictions for {image_path}:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {self.class_names[idx]}: {pred_probs[idx]:.4f}")
        
        return pred_probs, self.class_names[top_indices[0]]
    
    def save_model(self, filepath="palm_disease_classifier.keras", format_type="keras"):
        """
        Save the trained model in specified format
        
        Args:
            filepath (str): Path to save the model
            format_type (str): 'keras' (recommended) or 'h5' (legacy)
        """
        if self.model is not None:
            if format_type.lower() == "keras":
                # Modern Keras format (recommended)
                if not filepath.endswith('.keras'):
                    filepath = filepath.replace('.h5', '.keras')
                self.model.save(filepath)
                print(f"Model saved in Keras format to {filepath}")
                
            elif format_type.lower() == "h5":
                # Legacy H5 format
                if not filepath.endswith('.h5'):
                    filepath = filepath.replace('.keras', '.h5')
                self.model.save(filepath, save_format='h5')
                print(f"Model saved in H5 format to {filepath}")
                
            else:
                raise ValueError("format_type must be 'keras' or 'h5'")
                
            # Also save model architecture as JSON (for compatibility)
            model_json = self.model.to_json()
            json_filepath = filepath.replace('.keras', '_architecture.json').replace('.h5', '_architecture.json')
            with open(json_filepath, 'w') as json_file:
                json_file.write(model_json)
            print(f"Model architecture saved to {json_filepath}")
            
        else:
            print("No model to save!")
    
    def load_model(self, filepath, format_type="auto"):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
            format_type (str): 'auto', 'keras', or 'h5'
        """
        if format_type == "auto":
            # Auto-detect format based on file extension
            if filepath.endswith('.keras'):
                format_type = "keras"
            elif filepath.endswith('.h5'):
                format_type = "h5"
            else:
                raise ValueError("Cannot auto-detect format. Specify format_type='keras' or 'h5'")
        
        try:
            if format_type.lower() == "keras":
                self.model = keras.models.load_model(filepath)
                print(f"Model loaded from Keras format: {filepath}")
            elif format_type.lower() == "h5":
                self.model = keras.models.load_model(filepath)
                print(f"Model loaded from H5 format: {filepath}")
            else:
                raise ValueError("format_type must be 'keras' or 'h5'")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Try specifying the correct format_type or check file path")

def main():
    """
    Main training pipeline
    """
    # Configuration
    DATA_DIR = "path/to/your/palm_disease_dataset"  # Update this path
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # Optional: Standardize file extensions before training
    print("Checking for uppercase file extensions...")
    PalmDiseaseClassifier.standardize_extensions(DATA_DIR, dry_run=True)
    
    # Uncomment the line below to actually rename files with uppercase extensions
    # PalmDiseaseClassifier.standardize_extensions(DATA_DIR, dry_run=False)
    
    # Initialize classifier
    classifier = PalmDiseaseClassifier(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Load and prepare data
    train_data, val_data, test_data = classifier.load_and_prepare_data()
    
    # Train model
    history = classifier.train_model(train_data, val_data, epochs=EPOCHS)
    
    # Evaluate model
    test_acc, test_top2_acc = classifier.evaluate_model(test_data)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save model in both formats for maximum compatibility
    classifier.save_model("final_palm_disease_model.keras", format_type="keras")  # Recommended
    classifier.save_model("final_palm_disease_model.h5", format_type="h5")        # Legacy backup
    
    print("\nTraining completed successfully!")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Top-2 Accuracy: {test_top2_acc:.4f}")
    print("Models saved in both Keras (.keras) and H5 (.h5) formats")
    
    # Example prediction
    # classifier.predict_single_image("path/to/test/image.jpg")

# Example usage for loading models:
def load_and_predict_example():
    """
    Example of how to load saved models and make predictions
    """
    classifier = PalmDiseaseClassifier("path/to/dataset")
    
    # Load Keras format (recommended)
    classifier.load_model("final_palm_disease_model.keras")
    
    # Or load H5 format
    # classifier.load_model("final_palm_disease_model.h5")
    
    # Make prediction
    classifier.predict_single_image("path/to/test/image.jpg")

if __name__ == "__main__":
    main()