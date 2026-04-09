# ============================================
# YOLOv8 GTSRB Training - COMPLETE FIXED VERSION
# ============================================
import os
import pandas as pd
import shutil
from PIL import Image
from ultralytics import YOLO
import numpy as np


# ============================================
# 2. SET YOUR DATASET PATH
# ============================================
BASE_PATH = 'E:/Project/gtsrb'  # Update if different

# ============================================
# 3. LOAD AND INSPECT DATASETS
# ============================================
print("=== Loading Datasets ===")
train_df = pd.read_csv(f'{BASE_PATH}/Train.csv')
test_df = pd.read_csv(f'{BASE_PATH}/Test.csv')
meta_df = pd.read_csv(f'{BASE_PATH}/Meta.csv')

print(f"\n📊 Dataset Info:")
print(f"  Train samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Unique classes: {train_df['ClassId'].nunique()}")

print(f"\n📋 Train.csv columns: {train_df.columns.tolist()}")
print(f"📋 Test.csv columns: {test_df.columns.tolist()}")
print(f"📋 Meta.csv columns: {meta_df.columns.tolist()}")

print(f"\n🔍 Meta.csv content:")
print(meta_df.head(10))

# ============================================
# 4. GET CLASS NAMES (FIXED)
# ============================================
# Get number of classes
num_classes = train_df['ClassId'].nunique()

# Try to get class names from Meta.csv (handle different column name possibilities)
possible_name_columns = ['Name', 'SignName', 'ClassName', 'ClassDescription', 'name', 'sign_name']

class_names = None
for col in possible_name_columns:
    if col in meta_df.columns:
        print(f"\n✅ Found class names in column: '{col}'")
        class_names = meta_df.sort_values('ClassId')[col].tolist()
        break

# If no name column found, create generic names
if class_names is None:
    print(f"\n⚠️ No name column found in Meta.csv. Using generic class names.")
    class_names = [f'traffic_sign_class_{i}' for i in range(num_classes)]
else:
    # Ensure we have the right number of class names
    if len(class_names) != num_classes:
        print(f"\n⚠️ Class name count mismatch. Creating {num_classes} generic names.")
        class_names = [f'traffic_sign_class_{i}' for i in range(num_classes)]

print(f"\n✅ Total classes: {num_classes}")
print(f"✅ First 10 class names: {class_names[:10]}")

# ============================================
# 5. CREATE YOLO DIRECTORY STRUCTURE
# ============================================
YOLO_BASE = 'E:/Project/gtsrb/gtsrb_yolo'

print(f"\n📁 Creating YOLO directory structure at: {YOLO_BASE}")

for split in ['train', 'val', 'test']:
    os.makedirs(f'{YOLO_BASE}/images/{split}', exist_ok=True)
    os.makedirs(f'{YOLO_BASE}/labels/{split}', exist_ok=True)

print("✅ Directory structure created!")

# ============================================
# 6. CONVERSION FUNCTION
# ============================================
def convert_gtsrb_to_yolo(df, source_folder, output_split):
    """
    Convert GTSRB format to YOLOv8 format

    GTSRB CSV: Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
    YOLO: class_id x_center y_center width height (all normalized 0-1)
    """
    print(f"\n🔄 Converting {output_split} dataset...")

    success_count = 0
    error_count = 0
    error_messages = []

    for idx, row in df.iterrows():
        try:
            # Build image path - handle different path formats
            image_path = row['Path']

            # Try different path combinations
            possible_paths = [
                os.path.join(BASE_PATH, source_folder, image_path),
                os.path.join(BASE_PATH, image_path),
                os.path.join(BASE_PATH, source_folder, os.path.basename(image_path))
            ]

            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break

            if img_path is None:
                error_count += 1
                if error_count <= 3:
                    error_messages.append(f"Image not found: {image_path}")
                continue

            # Get bounding box info
            img_width = int(row['Width'])
            img_height = int(row['Height'])
            x1 = float(row['Roi.X1'])
            y1 = float(row['Roi.Y1'])
            x2 = float(row['Roi.X2'])
            y2 = float(row['Roi.Y2'])
            class_id = int(row['ClassId'])

            # Convert to YOLO format (normalized coordinates)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            bbox_width = (x2 - x1) / img_width
            bbox_height = (y2 - y1) / img_height

            # Clip to valid range [0, 1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            bbox_width = np.clip(bbox_width, 0, 1)
            bbox_height = np.clip(bbox_height, 0, 1)

            # Skip invalid boxes
            if bbox_width <= 0 or bbox_height <= 0:
                error_count += 1
                continue

            # Generate output filenames
            img_filename = f"{output_split}_{idx:06d}.png"
            label_filename = f"{output_split}_{idx:06d}.txt"

            # Copy image
            output_img_path = f'{YOLO_BASE}/images/{output_split}/{img_filename}'
            shutil.copy(img_path, output_img_path)

            # Create YOLO label file
            label_path = f'{YOLO_BASE}/labels/{output_split}/{label_filename}'
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

            success_count += 1

            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"  ⏳ Processed {idx + 1}/{len(df)} images...")

        except Exception as e:
            error_count += 1
            if len(error_messages) < 3:
                error_messages.append(f"Error at row {idx}: {str(e)}")

    print(f"✅ {output_split}: {success_count} images converted successfully")
    if error_count > 0:
        print(f"⚠️  {error_count} images failed")
        if error_messages:
            print("  First few errors:")
            for msg in error_messages[:3]:
                print(f"    - {msg}")

    return success_count, error_count

# ============================================
# 7. SPLIT TRAINING DATA (80/20)
# ============================================
from sklearn.model_selection import train_test_split

print("\n📊 Splitting training data...")

train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df['ClassId']
)

print(f"  Training set: {len(train_data)} images")
print(f"  Validation set: {len(val_data)} images")
print(f"  Test set: {len(test_df)} images")

# ============================================
# 8. CONVERT ALL DATASETS
# ============================================
print("\n🔄 Converting datasets to YOLO format...")

# Convert train split
train_success, train_errors = convert_gtsrb_to_yolo(train_data, 'Train', 'train')

# Convert validation split
val_success, val_errors = convert_gtsrb_to_yolo(val_data, 'Train', 'val')

# Convert test set
test_success, test_errors = convert_gtsrb_to_yolo(test_df, 'Test', 'test')

# ============================================
# 9. CREATE DATASET.YAML
# ============================================
yaml_content = f"""# YOLOv8 Dataset Configuration for GTSRB
path: {YOLO_BASE}
train: images/train
val: images/val
test: images/test

# Classes
nc: {num_classes}
names: {class_names}
"""

yaml_path = 'E:/Project/gtsrb/gtsrb.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n✅ Dataset configuration saved to: {yaml_path}")

# ============================================
# 10. VERIFY CONVERSION
# ============================================
print("\n🔍 Verifying converted dataset...")

train_img_count = len(os.listdir(f'{YOLO_BASE}/images/train'))
train_lbl_count = len(os.listdir(f'{YOLO_BASE}/labels/train'))
val_img_count = len(os.listdir(f'{YOLO_BASE}/images/val'))
val_lbl_count = len(os.listdir(f'{YOLO_BASE}/labels/val'))
test_img_count = len(os.listdir(f'{YOLO_BASE}/images/test'))
test_lbl_count = len(os.listdir(f'{YOLO_BASE}/labels/test'))

print(f"  Train: {train_img_count} images, {train_lbl_count} labels")
print(f"  Val: {val_img_count} images, {val_lbl_count} labels")
print(f"  Test: {test_img_count} images, {test_lbl_count} labels")

# Check sample label
if train_lbl_count > 0:
    sample_label = os.listdir(f'{YOLO_BASE}/labels/train')[0]
    print(f"\n📄 Sample label file ({sample_label}):")
    with open(f'{YOLO_BASE}/labels/train/{sample_label}', 'r') as f:
        print(f"  {f.read().strip()}")

# ============================================
# 11. LOAD YOLOv8 MODEL
# ============================================
print("\n🤖 Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # Nano version for faster training
print("✅ YOLOv8n model loaded!")

# ============================================
# 12. TRAIN THE MODEL
# ============================================
import torch
# ============================================
# 1. GPU/CPU AUTO-DETECTION (FIXED)
# ============================================
print("="*60)
print("🖥️  SYSTEM CHECK")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# IMPORTANT: Check BOTH cuda.is_available() AND device_count()
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    device = 0  # Use GPU
    batch_size = 16
    workers = 4
    epochs = 50
    print("✅ Using GPU for training")
else:
    device = 'cpu'  # Use CPU
    batch_size = 4
    workers = 2
    epochs = 10
    print("⚠️  NO GPU DETECTED - Using CPU")
    print("⚠️  Training will be VERY SLOW (8-12 hours)")
    print("\n💡 To enable GPU:")
    print("   1. Runtime → Change runtime type")
    print("   2. Hardware accelerator → T4 GPU")
    print("   3. Save and restart")

print(f"\n⚙️  Configuration:")
print(f"   Device: {device}")
print(f"   Batch size: {batch_size}")
print(f"   Workers: {workers}")
print(f"   Epochs: {epochs}")
print("="*60)
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print(f"Device: {device}")
print(f"Batch: {batch_size}")
print(f"Epochs: {epochs}")

if device == 'cpu':
    print("\n⚠️  WARNING: Training on CPU (SLOW!)")
    print("Expected time: 8-12 hours")
    print("Consider enabling GPU for 1-2 hour training")

print("="*60 + "\n")

results = model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=640,
    batch=batch_size,
    name='gtsrb_yolov8',
    patience=10,
    device=device,  # ← FIXED: Use the variable, not hardcoded 0
    workers=workers,
    plots=True,
    save=True,
    project='E:/Project/gtsrb/runs/detect',
    exist_ok=True,
    verbose=True,
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    amp=True if device == 0 else False,  # This is fine
    seed=42
)

print("\n✅ TRAINING COMPLETE!")

# ============================================
# 13. VALIDATION
# ============================================
print("\n📊 Validating model...")
val_metrics = model.val()

print(f"\n📈 Validation Metrics:")
print(f"  mAP50: {val_metrics.box.map50:.4f}")
print(f"  mAP50-95: {val_metrics.box.map:.4f}")
print(f"  Precision: {val_metrics.box.mp:.4f}")
print(f"  Recall: {val_metrics.box.mr:.4f}")

# ============================================
# 14. TEST ON TEST SET
# ============================================
print("\n🧪 Testing on test set...")
test_metrics = model.val(data=yaml_path, split='test')

print(f"\n📈 Test Metrics:")
print(f"  mAP50: {test_metrics.box.map50:.4f}")
print(f"  mAP50-95: {test_metrics.box.map:.4f}")
print(f"  Precision: {test_metrics.box.mp:.4f}")
print(f"  Recall: {test_metrics.box.mr:.4f}")

# ============================================
# 15. VISUALIZE RESULTS
# ============================================
from IPython.display import Image, display

print("\n📊 Visualizing Results...")

results_dir = 'E:/Project/gtsrb/runs/detect/gtsrb_yolov8'

# Training curves
if os.path.exists(f'{results_dir}/results.png'):
    print("\n📈 Training Curves:")
    display(Image(filename=f'{results_dir}/results.png', width=1000))

# Confusion matrix
if os.path.exists(f'{results_dir}/confusion_matrix.png'):
    print("\n🔲 Confusion Matrix:")
    display(Image(filename=f'{results_dir}/confusion_matrix.png', width=800))

# Sample predictions
if os.path.exists(f'{results_dir}/val_batch0_pred.jpg'):
    print("\n🖼️ Sample Validation Predictions:")
    display(Image(filename=f'{results_dir}/val_batch0_pred.jpg', width=1000))

# ============================================
# 16. TEST PREDICTIONS
# ============================================
print("\n🔍 Running predictions on test images...")

test_images = os.listdir(f'{YOLO_BASE}/images/test')[:5]

for img_name in test_images:
    img_path = f'{YOLO_BASE}/images/test/{img_name}'

    results = model.predict(
        source=img_path,
        conf=0.25,
        save=True,
        project='E:/Project/gtsrb/test_predictions',
        name='results',
        exist_ok=True
    )

    # Display
    result_path = f'E:/Project/gtsrb/test_predictions/results/{img_name}'
    if os.path.exists(result_path):
        print(f"\n🖼️ {img_name}:")
        display(Image(filename=result_path, width=600))

        # Print detections
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    print(f"  ✓ {cls_name}: {conf:.2%}")

# ============================================
# 17. EXPORT MODEL
# ============================================
print("\n💾 Exporting model...")
onnx_path = model.export(format='onnx')
print(f"✅ ONNX model exported to: {onnx_path}")

# ============================================
# 18. SAVE TO GOOGLE DRIVE
# ============================================
print("\n💾 Saving models to folder...")

drive_output = 'E:/Project/gtsrb/gtsrb_yolov8_trained'
os.makedirs(drive_output, exist_ok=True)


   
shutil.copy(f"{results_dir}/weights/best.pt", f"{drive_output}/best.pt")
shutil.copy(f"{results_dir}/weights/last.pt", f"{drive_output}/last.pt")
shutil.copy(onnx_path, f"{drive_output}/model.onnx")
shutil.copytree(results_dir, f"{drive_output}/training_results", dirs_exist_ok=True)


print(f"✅ Models saved to: {drive_output}")
# ============================================
# 19. FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE - SUMMARY")
print("="*60)
print(f"✅ Model: YOLOv8n")
print(f"✅ Dataset: GTSRB")
print(f"✅ Classes: {num_classes}")
print(f"✅ Train images: {train_img_count}")
print(f"✅ Val images: {val_img_count}")
print(f"✅ Test images: {test_img_count}")
print(f"\n📊 Performance:")
print(f"  Val mAP50: {val_metrics.box.map50:.4f}")
print(f"  Test mAP50: {test_metrics.box.map50:.4f}")
print(f"\n💾 Saved to: {drive_output}")
print("="*60)
