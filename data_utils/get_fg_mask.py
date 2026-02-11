import os
import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import argparse

# Label map of TotalSegmentator
label_map = {
    'spleen': 1,
    'kidney_right': 2,
    'kidney_left': 3,
    'gallbladder': 4,
    'liver': 5,
    'stomach': 6,
    'pancreas': 7,
    'adrenal_gland_right': 8,
    'adrenal_gland_left': 9,
    'lung_upper_lobe_left': 10,
    'lung_lower_lobe_left': 11,
    'lung_upper_lobe_right': 12,
    'lung_middle_lobe_right': 13,
    'lung_lower_lobe_right': 14,
    'esophagus': 15,
    'trachea': 16,
    'thyroid_gland': 17,
    'small_bowel': 18,
    'duodenum': 19,
    'colon': 20,
    'urinary_bladder': 21,
    'prostate': 22,
    'kidney_cyst_left': 23,
    'kidney_cyst_right': 24,
    'sacrum': 25,
    'vertebrae_S1': 26,
    'vertebrae_L5': 27,
    'vertebrae_L4': 28,
    'vertebrae_L3': 29,
    'vertebrae_L2': 30,
    'vertebrae_L1': 31,
    'vertebrae_T12': 32,
    'vertebrae_T11': 33,
    'vertebrae_T10': 34,
    'vertebrae_T9': 35,
    'vertebrae_T8': 36,
    'vertebrae_T7': 37,
    'vertebrae_T6': 38,
    'vertebrae_T5': 39,
    'vertebrae_T4': 40,
    'vertebrae_T3': 41,
    'vertebrae_T2': 42,
    'vertebrae_T1': 43,
    'vertebrae_C7': 44,
    'vertebrae_C6': 45,
    'vertebrae_C5': 46,
    'vertebrae_C4': 47,
    'vertebrae_C3': 48,
    'vertebrae_C2': 49,
    'vertebrae_C1': 50,
    'heart': 51,
    'aorta': 52,
    'pulmonary_vein': 53,
    'brachiocephalic_trunk': 54,
    'subclavian_artery_right': 55,
    'subclavian_artery_left': 56,
    'common_carotid_artery_right': 57,
    'common_carotid_artery_left': 58,
    'brachiocephalic_vein_left': 59,
    'brachiocephalic_vein_right': 60,
    'atrial_appendage_left': 61,
    'superior_vena_cava': 62,
    'inferior_vena_cava': 63,
    'portal_vein_and_splenic_vein': 64,
    'iliac_artery_left': 65,
    'iliac_artery_right': 66,
    'iliac_vena_left': 67,
    'iliac_vena_right': 68,
    'humerus_left': 69,
    'humerus_right': 70,
    'scapula_left': 71,
    'scapula_right': 72,
    'clavicula_left': 73,
    'clavicula_right': 74,
    'femur_left': 75,
    'femur_right': 76,
    'hip_left': 77,
    'hip_right': 78,
    'spinal_cord': 79,
    'gluteus_maximus_left': 80,
    'gluteus_maximus_right': 81,
    'gluteus_medius_left': 82,
    'gluteus_medius_right': 83,
    'gluteus_minimus_left': 84,
    'gluteus_minimus_right': 85,
    'autochthon_left': 86,
    'autochthon_right': 87,
    'iliopsoas_left': 88,
    'iliopsoas_right': 89,
    'brain': 90,
    'skull': 91,
    'rib_left_1': 92,
    'rib_left_2': 93,
    'rib_left_3': 94,
    'rib_left_4': 95,
    'rib_left_5': 96,
    'rib_left_6': 97,
    'rib_left_7': 98,
    'rib_left_8': 99,
    'rib_left_9': 100,
    'rib_left_10': 101,
    'rib_left_11': 102,
    'rib_left_12': 103,
    'rib_right_1': 104,
    'rib_right_2': 105,
    'rib_right_3': 106,
    'rib_right_4': 107,
    'rib_right_5': 108,
    'rib_right_6': 109,
    'rib_right_7': 110,
    'rib_right_8': 111,
    'rib_right_9': 112,
    'rib_right_10': 113,
    'rib_right_11': 114,
    'rib_right_12': 115,
    'sternum': 116,
    'costal_cartilages': 117
}

# Radiologist region labels ROI
radiologist_region_labels_roi = {
    "splenomegaly": [
        "spleen"
    ],
    "adrenal_hyperplasia": [
        "adrenal_gland_left",
        "adrenal_gland_right"
    ],
    "fatty_liver": [
        "liver"
    ],
    "cholecystitis": [
        "gallbladder",
        "liver"
    ],
    "liver_calcifications": [
        "liver"
    ],
    "hydronephrosis": [
        "kidney_left",
        "kidney_right"
    ],
    "gallstone": [
        "gallbladder"
    ],
    "liver_lesion": [
        "liver"
    ],
    "kidney_stone": [
        "kidney_left",
        "kidney_right"
    ],
    "liver_cyst": [
        "liver"
    ],
    "renal_cyst": [
        "kidney_cyst_left",
        "kidney_cyst_right",
        "kidney_left",
        "kidney_right"
    ]
}

non_roi_diseases = [
    "atherosclerosis",
    "colorectal_cancer",
    "ascites",
    "lymphadenopathy"
]


def create_fg_mask(label_img, fg_label_values):
    """
    Create a binary foreground mask from a label image.

    Args:
        label_img: SimpleITK image with label values
        fg_label_values: List of label values to consider as foreground

    Returns:
        SimpleITK image with binary mask (1 for foreground, 0 for background)
    """
    label_array = sitk.GetArrayFromImage(label_img)

    # Create binary mask
    fg_mask = np.zeros_like(label_array, dtype=np.uint8)
    for label_val in fg_label_values:
        fg_mask[label_array == label_val] = 1

    # Convert back to SimpleITK image
    fg_mask_img = sitk.GetImageFromArray(fg_mask)
    fg_mask_img.CopyInformation(label_img)

    return fg_mask_img


def process_single_case(args):
    """
    Process a single case - worker function for multiprocessing.

    Args:
        args: Tuple of (case_id, label_path, output_path, fg_label_values)

    Returns:
        Tuple of (case_id, success, error_msg)
    """
    case_id, label_path, output_path, fg_label_values = args

    try:
        if not os.path.exists(label_path):
            return (case_id, False, "Label file not found")

        # Load the label image
        label_img = sitk.ReadImage(label_path)

        # Create foreground mask
        fg_mask_img = create_fg_mask(label_img, fg_label_values)

        # Save the foreground mask
        sitk.WriteImage(fg_mask_img, output_path)

        return (case_id, True, None)

    except Exception as e:
        return (case_id, False, str(e))


def process_disease(disease_name, csv_path, labels_dir, output_dir, num_workers=None):
    """
    Process a single disease CSV and create foreground masks for all cases.

    Args:
        disease_name: Name of the disease
        csv_path: Path to the disease CSV file
        labels_dir: Directory containing TotalSeg labels
        output_dir: Directory to save foreground masks
        num_workers: Number of parallel workers (default: CPU count)
    """
    # Check if this disease has ROI labels
    if disease_name in non_roi_diseases:
        print(f"Skipping {disease_name} - no ROI-based labels defined")
        return

    if disease_name not in radiologist_region_labels_roi:
        print(f"Warning: {disease_name} not found in radiologist_region_labels_roi, skipping")
        return

    # Get the organ names for this disease
    organ_names = radiologist_region_labels_roi[disease_name]

    # Convert organ names to label values
    fg_label_values = [label_map[organ] for organ in organ_names]

    print(f"\nProcessing {disease_name}")
    print(f"  Organs: {organ_names}")
    print(f"  Label values: {fg_label_values}")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create output directory for this disease
    disease_output_dir = os.path.join(output_dir, disease_name)
    os.makedirs(disease_output_dir, exist_ok=True)

    # Prepare arguments for all cases
    total_cases = len(df)
    args_list = []

    for _, row in df.iterrows():
        case_id = row['case_id']
        label_path = os.path.join(labels_dir, case_id)
        output_path = os.path.join(disease_output_dir, case_id)
        args_list.append((case_id, label_path, output_path, fg_label_values))

    # Set number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"  Processing {total_cases} cases with {num_workers} workers...")

    # Process cases in parallel
    processed = 0
    skipped = 0
    errors = []

    with mp.Pool(num_workers) as pool:
        # Use imap_unordered for better memory efficiency and progress tracking
        for i, (case_id, success, error_msg) in enumerate(pool.imap_unordered(process_single_case, args_list), 1):
            if success:
                processed += 1
            else:
                skipped += 1
                errors.append((case_id, error_msg))

            # Print progress every 50 cases
            if i % 50 == 0:
                print(f"  Progress: {i}/{total_cases} cases...")

    print(f"  Completed: {processed} processed, {skipped} skipped")

    if errors:
        print(f"  Errors encountered:")
        for case_id, error_msg in errors[:10]:  # Show first 10 errors
            print(f"    {case_id}: {error_msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")


def main():
    # Define paths
    parser = argparse.ArgumentParser(description="Create foreground masks from label images")
    parser.add_argument("--csv-dir", type=str, default="/path/to/amos-clf-tr-val/labels", help="Directory containing disease CSV files")
    parser.add_argument("--labels-dir", type=str, default="/path/to/amos-clf-tr-val/totalseg", help="Directory containing TotalSeg labels")
    parser.add_argument("--output-dir", type=str, default="/path/to/amos-clf-tr-val/fg_masks", help="Directory to save foreground masks")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    csv_dir = args.csv_dir
    labels_dir = args.labels_dir
    output_dir = args.output_dir
    num_workers = args.num_workers
    # Number of parallel workers (None = use all CPUs)
    num_workers = None  # Change to specific number if needed, e.g., 16

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all disease CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    print(f"Found {len(csv_files)} disease CSV files")
    print(f"Labels directory: {labels_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using {num_workers if num_workers else mp.cpu_count()} workers\n")

    # Process each disease
    for csv_path in sorted(csv_files):
        disease_name = os.path.basename(csv_path).replace('.csv', '')
        process_disease(disease_name, csv_path, labels_dir, output_dir, num_workers)

    print("\nAll done!")


if __name__ == "__main__":
    main()
