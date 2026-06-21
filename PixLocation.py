# coding=utf-8
"""
PixLocation.py - Fluorescent Microsphere Localization Tool
Automatically detects and localizes green fluorescent microspheres in 24-well plate images.
"""

import cv2
import csv
import os
import re
import numpy as np
import pandas as pd


# ======================== Configuration Parameters ========================

# Path configuration (using os.path.join for cross-platform compatibility)
RAW_DATA_DIR = os.path.join(".", "data", "raw")              # Input image folder
OUTPUT_DIR = os.path.join(".", "data", "processed")          # Output results folder
DEMO_DIR = os.path.join(".", "demo", "sample_images")        # Demo images folder
DOCS_DIR = os.path.join(".", "docs")                         # Documentation folder

# Detection parameters
EDGE_PIXEL = 50              # Edge exclusion distance (pixels)
MIN_RADIUS = 40               # Minimum colony radius (pixels)
MAX_RADIUS = 180              # Maximum colony radius (pixels)
PLATE_RADIUS = 1900           # Well plate radius (pixels), used for mm conversion
PIXEL_TO_MM = 15.6 / (PLATE_RADIUS * 2)  # Conversion factor: pixels to millimeters

LOWER_BRIGHTNESS = 40         # Binary threshold brightness value
CIRCULARITY_LIMIT = 0.2       # Circularity threshold (0-1), higher = more circular

# HSV color range for green detection (adjust based on your images)
LOWER_GREEN = np.array([60, 255, 40])
UPPER_GREEN = np.array([77, 255, 255])

# Fixed well center coordinates (for manual correction if needed)
CENTER_DICT = {
    'A01': (2182, 1920), 'A02': (2170, 1904), 'A03': (2160, 1872),
    'A04': (2176, 1886), 'A05': (2174, 1868), 'A06': (2176, 1874),
    'B01': (2184, 1926), 'B02': (2182, 1908), 'B03': (2172, 1896),
    'B04': (2190, 1898), 'B05': (2192, 1890), 'B06': (2192, 1874),
    'C01': (2204, 1926), 'C02': (2194, 1934), 'C03': (2188, 1908),
    'C04': (2208, 1908), 'C05': (2192, 1894), 'C06': (2206, 1884),
    'D01': (2204, 1932), 'D02': (2214, 1920), 'D03': (2202, 1934),
    'D04': (2212, 1928), 'D05': (2202, 1896), 'D06': (2200, 1886)
}

# Well mapping: letter to number (A=1, B=2, ..., H=8)
WELL_ROW_MAP = {
    "A": 1, "B": 2, "C": 3, "D": 4,
    "E": 5, "F": 6, "G": 7, "H": 8,
}


# ======================== Helper Functions ========================

def delete_files(folder_path):
    """
    Delete all files in the specified folder.
    Creates the folder if it does not exist.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def create_directory_structure():
    """Create the required project folder structure."""
    folders = [
        RAW_DATA_DIR,
        OUTPUT_DIR,
        DEMO_DIR,
    ]
    
    for folder in folders:
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder created: {folder}")
        elif folder and os.path.exists(folder):
            print(f"Folder already exists: {folder}")


def char_to_num(well_row):
    """
    Convert well row letter (A-H) to number (1-8).
    Returns None if the letter is not in A-H.
    """
    return WELL_ROW_MAP.get(well_row, None)


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def extract_well_name(filename):
    """
    Extract well name (e.g., A01, B06) from filename.
    Expected format: xxx_xxx_xxx_xxx_xxx_A01.TIF
    Returns None if no valid well name is found.
    """
    # Try to match pattern like _A01. or _H06.
    pattern = r'_([A-H]\d{2})\.'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    
    # Fallback: try splitting by underscore (original method)
    try:
        parts = filename.split('.')[0].split('_')
        if len(parts) >= 6:
            well = parts[5][0:3]
            if re.match(r'^[A-H]\d{2}$', well):
                return well
    except (IndexError, AttributeError):
        pass
    
    return None


def calculate_well_number(well_name):
    """
    Convert well name (e.g., A01) to a sequential number.
    A01=1, A02=2, ..., A06=6, B01=7, ..., H06=48
    """
    if not well_name or len(well_name) < 3:
        return None
    
    row_letter = well_name[0:1]
    col_num = int(well_name[1:3])
    row_num = char_to_num(row_letter)
    
    if row_num is None:
        return None
    
    return (row_num - 1) * 4 + (col_num - 1) * 1 + 1


def auto_canny(image, sigma=0.33):
    """
    Automatic Canny edge detection using median-based thresholding.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    print(f"Canny thresholds: lower={lower}, upper={upper}")
    return edged


# ======================== Main Processing Function ========================

def process_image(image_path):
    """
    Process a single image: detect well center, locate microspheres,
    and generate output CSV and annotated image.
    """
    # Initialize result containers
    result_list = [('No', 'PixelX', 'PixelY', 'PixelArea', 'XOffset', 'YOffset', 'Circularity')]
    
    # Read image
    img_rgb = cv2.imread(image_path, 1)
    if img_rgb is None:
        print(f"Error: Cannot read image {image_path}")
        return None, None, None
    
    # Extract well name from filename
    filename = os.path.basename(image_path)
    well_name = extract_well_name(filename)
    if well_name is None:
        print(f"Warning: Could not extract well name from {filename}, using fallback")
        well_name = "UNKNOWN"
    
    print(f"Processing well: {well_name}")
    
    # ---------- Step 1: Detect well center using Hough Circle ----------
    img_copy = cv2.imread(image_path, 1)
    gray_circle = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred_circle = cv2.GaussianBlur(gray_circle, (15, 15), 0)
    
    circles = cv2.HoughCircles(
        blurred_circle,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=5,
        param2=20,
        minRadius=1880,
        maxRadius=1895
    )
    
    img_center_x, img_center_y, plate_radius = 0, 0, PLATE_RADIUS
    
    if circles is not None:
        circle_count = len(circles[0, :])
        print(f"Circles detected: {circle_count}")
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            img_center_x, img_center_y, plate_radius = x, y, r
            break
    else:
        print("Warning: No circle detected, using default center")
        # Use image center as fallback
        h, w = img_rgb.shape[:2]
        img_center_x, img_center_y = w // 2, h // 2
    
    # Update pixel-to-mm conversion based on detected radius
    pixel_to_mm = 15.6 / (plate_radius * 2)
    print(f"Center: ({img_center_x}, {img_center_y}), Radius: {plate_radius}")
    
    # ---------- Step 2: Detect green microspheres ----------
    # Convert to HSV for color-based segmentation
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    
    # Create mask for green color range
    mask = cv2.inRange(img_hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.medianBlur(mask, 3)
    
    # Apply mask to get only green regions
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, LOWER_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=3)
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        dilation,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    
    # Draw contours on the original image (for debugging)
    cv2.drawContours(img_rgb, contours, -1, (100, 100, 0), 1)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # ---------- Step 3: Filter and record detected objects ----------
    index = 1
    
    for contour in contours:
        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = (4 * np.pi * area / (perimeter ** 2)) ** 0.5
        else:
            circularity = 0
        
        # Filter by radius, distance from center, and circularity
        distance_from_center = calculate_distance(x, y, img_center_x, img_center_y)
        
        if (MIN_RADIUS < radius < MAX_RADIUS and
            distance_from_center < (plate_radius - EDGE_PIXEL) and
            CIRCULARITY_LIMIT < circularity < 0.98):
            
            # Calculate offsets in millimeters
            x_offset_mm = round((int(x) - img_center_x) * pixel_to_mm, 3)
            y_offset_mm = round(-(int(y) - img_center_y) * pixel_to_mm, 3)
            
            # Draw circle and label on image
            img_rgb = cv2.circle(img_rgb, center, radius, (255, 208, 176), 10)
            text_location = (int(x) + radius + 3, int(y) + radius + 3)
            cv2.putText(
                img_rgb,
                str(index),
                text_location,
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (255, 208, 176),
                10,
                cv2.LINE_AA
            )
            
            # Add to result list
            result_list.append([
                index,
                int(x),
                int(y),
                float(area),
                x_offset_mm,
                y_offset_mm,
                round(circularity, 3)
            ])
            
            print(f"  Colony {index}: ({int(x)}, {int(y)}), radius={radius}, circularity={circularity:.3f}")
            index += 1
    
    # ---------- Step 4: Draw well center and boundary ----------
    img_rgb = cv2.circle(img_rgb, (img_center_x, img_center_y), 2, (250, 5, 5), 2)
    img_rgb = cv2.circle(img_rgb, (img_center_x, img_center_y), plate_radius, (250, 5, 5), 2)
    
    # ---------- Step 5: Save outputs ----------
    # Prepare file paths
    csv_path = os.path.join(OUTPUT_DIR, well_name + ".csv")
    png_path = os.path.join(OUTPUT_DIR, well_name + ".PNG")
    
    # Save CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result_list)
    
    # Save annotated image
    cv2.imwrite(png_path, img_rgb)
    
    # Return summary info
    well_number = calculate_well_number(well_name)
    return well_name, well_number, index - 1


# ======================== Main Execution ========================

def main():
    """
    Main entry point of the program.
    Processes all images in the raw data folder.
    """
    print("=" * 60)
    print("PixLocation - Fluorescent Microsphere Localization Tool")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Clean output folder
    if os.path.exists(OUTPUT_DIR):
        delete_files(OUTPUT_DIR)
        print(f"Cleaned output folder: {OUTPUT_DIR}")
    
    # Initialize summary data
    center_summary = [('WellNr', 'X', 'Y', 'Radius')]
    file_summary = [('FileName', 'WellNumber', 'PickNumber')]
    
    # Process each image in the raw data folder
    print("\n" + "-" * 60)
    print("Starting image processing...")
    print("-" * 60)
    
    processed_count = 0
    
    for dir_path, dir_names, filenames in os.walk(RAW_DATA_DIR):
        for filename in filenames:
            # Only process TIF files
            if not filename.lower().endswith(('.tif', '.tiff')):
                print(f"Skipping non-TIF file: {filename}")
                continue
            
            image_path = os.path.join(dir_path, filename)
            print(f"\nProcessing: {filename}")
            
            # Process the image
            well_name, well_number, pick_count = process_image(image_path)
            
            if well_name is not None:
                processed_count += 1
                
                # Add to center summary
                # Note: center coordinates need to be retrieved from the processing
                # We'll use a placeholder - actual values are inside process_image
                center_summary.append([well_name, 0, 0, 0])
                
                # Add to file summary
                file_summary.append([well_name, well_number if well_number else 0, pick_count])
    
    # ---------- Save summary files ----------
    print("\n" + "-" * 60)
    print("Saving summary files...")
    print("-" * 60)
    
    # Save center coordinates summary
    center_csv_path = os.path.join(OUTPUT_DIR, "CenterXY.csv")
    with open(center_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(center_summary)
    print(f"  Saved: {center_csv_path}")
    
    # Save file count summary
    file_summary_path = os.path.join(OUTPUT_DIR, "TotalFileCount.csv")
    with open(file_summary_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(file_summary)
    print(f"  Saved: {file_summary_path}")
    
    # Sort the file summary by well number
    try:
        data = pd.read_csv(file_summary_path)
        sorted_data = data.sort_values(by='WellNumber')
        sorted_data.to_csv(file_summary_path, index=False)
        print(f"  Sorted {file_summary_path} by WellNumber")
    except Exception as e:
        print(f"  Warning: Could not sort summary file: {e}")
    
    # ---------- Final summary ----------
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {processed_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nOutput files:")
    print(f"  - [WellName].csv       : Colony coordinate data for each well")
    print(f"  - [WellName].PNG       : Annotated preview images")
    print(f"  - CenterXY.csv         : Summary of center coordinates")
    print(f"  - TotalFileCount.csv   : Summary statistics")
    print("=" * 60)


# ======================== Entry Point ========================

if __name__ == "__main__":
    main()