import os
from pathlib import Path
import shutil
import pandas as pd

def verify_dataset(data_path, strict=False):
    """
    Verify dataset structure with option for strict or flexible verification
    strict: If True, requires exact match with expected counts
           If False, allows more images than expected (for augmented data)
    """
    print("Verifying dataset structure...")
    total_images = 0
    expected_counts = {
        0: 93,    # سالم (Healthy)
        1: 684,   # کرم سفید ریشه (White Root Worm)
        2: 551,   # سفیدبالک (Whitefly)
        3: 183,   # شپشک آردالود (Mealybug)
        4: 1523,  # آبدزدک (Water Thief)
        5: 568,   # شته غلات (Grain Aphid)
        6: 304,   # آفت سبز (Green Pest)
        7: 590,   # شته یولاف (Oat Aphid)
        8: 913,   # زنجره (Leafhopper)
        9: 184,   # زنگ زدگی (Rust)
        10: 56,   # پوسیدگی (Rot)
        11: 108,  # لکه موجی (Wave Spot)
        12: 85,   # کپک (Mold)
        13: 197,  # بادزدگی (Wind Damage)
        14: 124,  # سفیدک پودری (Powdery Mildew)
        15: 721   # سایر (Others)
    }
    
    verification_passed = True
    
    for class_id in range(16):
        class_path = os.path.join(data_path, str(class_id))
        if not os.path.exists(class_path):
            print(f"Error: Directory for class {class_id} not found!")
            verification_passed = False
            continue
            
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        total_images += count
        
        expected = expected_counts[class_id]
        if strict and count != expected:
            verification_passed = False
        elif count < expected:  # In flexible mode, only fail if we have fewer images than expected
            verification_passed = False
            
        status = "✓" if (strict and count == expected) or (not strict and count >= expected) else "✗"
        print(f"Class {class_id}: Found {count} images (Expected: {expected}) {status}")
    
    print(f"\nTotal images: {total_images}")
    if not strict:
        print("Note: Additional images found in some classes (likely due to augmentation)")
        print("Verification passed if all classes have at least the minimum required images")
    
    return verification_passed

def prepare_test_dir(data_path):
    test_dir = os.path.join(data_path, "test_images")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    return test_dir

def verify_submission_format(file_path):
    """Verify if a submission file meets the challenge requirements"""
    # Check file name format
    file_name = os.path.basename(file_path)
    if not file_name.startswith('d_jabbari_7727_'):
        print(f"Error: Invalid file name format. Expected: d_jabbari_7727_N.csv, Got: {file_name}")
        return False
        
    # Check file extension
    if not file_name.endswith('.csv'):
        print(f"Error: File must be CSV format")
        return False
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Check columns
        required_columns = ['ID', 'class']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns: {required_columns}")
            return False
        
        # Check data types
        if not (df['ID'].dtype in ['int64', 'int32'] and df['class'].dtype in ['int64', 'int32']):
            print("Error: ID and class must be integers")
            return False
        
        # Check class values
        if not df['class'].between(0, 15).all():
            print("Error: class values must be between 0 and 15")
            return False
        
        # Check for duplicates
        if df['ID'].duplicated().any():
            print("Error: Duplicate IDs found")
            return False
        
        print(f"Submission format verification passed for {file_name}")
        print(f"Total predictions: {len(df)}")
        print("\nClass distribution:")
        print(df['class'].value_counts().sort_index())
        return True
        
    except Exception as e:
        print(f"Error verifying submission: {str(e)}")
        return False 