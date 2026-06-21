# AUTO-COMB-Platform
PixLocation.py - Fluorescent Microsphere Localization Tool
Project Overview
PixLocation.py is an OpenCV-based image processing tool for automatically identifying and locating green fluorescent microspheres in 24-well plates. The tool reads images captured with an EVOS7M7000 microscope, detects green targets, calculates their position coordinates, and generates CSV data files for use in downstream automated workflows.

## Features
- **Automatic detection of 24-well plate center positions
- **HSV color space-based green colony detection
- **Calculation of colony offset relative to well center (mm)
- **Circularity filtering to exclude non-target objects
- **Structured CSV data file output
- **Generation of annotated preview images

##  Requirements
- **Python 3.7+**
- **OpenCV 4.x** - `pip install opencv-python`
- **NumPy** - `pip install numpy`
- **Pandas** - `pip install pandas` 
- - **Pillow (PIL)** - `pip install pip install Pillow` 


##  Installation
1. Clone the repository
bash
git clone https://github.com/1332831738/AUTO-COMB-platform.git
cd AUTO-COMB-platform

2. Install dependencies
bash
pip install -r requirements.txt

Or install manually:
bash
pip install opencv-python numpy pandas pillow

3. Directory structure
AUTO-COMB-platform/
├── PixLocation.py                         # Main program entry
├── data/
│   ├── raw/                             # Input image folder (create manually)
│   │   └── *_*_*_*_&_A05*****.TIF         # TIF image files to be processed
│   │
│   └── processed/                       # Output folder (auto-generated)
│       ├── [WellName].csv               # Microsphere coordinate data for each well
│       ├── [WellName].PNG              # Annotated preview images for each well
│       ├── CenterXY.csv                  # Summary of center coordinates for all wells
│       └── TotalFileCount.csv             # Summary statistics file
├── demo/
│   └── sample_images/                  # Sample images for demonstration
├── LICENSE                             # MIT License
└── README.md                         # Project documentation

##  Usage

1. Prepare images
Place the image files to be analyzed into the data/raw/ folder. 
Recommended file naming format:
xxx_xxx_xxx_xxx_xxx_A01.TIF（Picture Resolution: 4408 x 3804:）
Here, A01 denotes the well position (rows A–H, columns 01–06), which the program will automatically extract.

2. Run the program
bash
python PixLocation.py

3. View output results
Once the program finishes running, the following will be generated in the data/processed/ folder:
CSV data files (one per well)
Containing the fields:

No: colony index
PixelX: pixel X coordinate
PixelY: pixel Y coordinate
PixelArea: pixel area
XOffset: X offset relative to center (mm)
YOffset: Y offset relative to center (mm)
Circularity: circularity (0–1)

Annotated images
Displaying well boundaries, center points, and detected colonies, with each colony labeled by an index number.

Summary files
CenterXY.csv: center coordinates for all wells
TotalFileCount.csv: number of colonies detected per well

Parameter Adjustment

To adjust detection parameters, modify the following variables in PixLocation.py:
Parameter   Default Value   Description
min_radius   10  Minimum colony radius (pixels)
max_radius   80   Maximum colony radius (pixels)
lower_Brightness  60   Binarization brightness 
thresholdCircularity_Limit  0.8  Circularity threshold (0–1)
lower_Green    [60, 255, 255]   HSV green range lower bound
upper_Green    [77, 255, 255]   HSV green range upper bound
Edge_Pixel    380    Edge exclusion distance (pixels)

Adjusting color range
To detect other colors, modify the HSV thresholds:
python# Example: red detection
lower_Green = np.array([156, 100, 100])
upper_Green = np.array([180, 255, 255])

Output Examples
Single well CSV output
csvNo,PixelX,PixelY,PixelArea,XOffset,YOffset,Circularity
1,2150,1930,254,0.156,-0.328,0.92
2,2180,1890,189,0.234,0.156,0.87
Summary statistics output
csvFileName,WellNumber,PickNumber
A01,1,2
A02,2,3
B01,5,1

##  Other important notes

Image quality: Ensure images are clear with good contrast and resolution is 4408 x 3804
Well alignment: The program automatically detects well centers, but it is recommended that wells be fully visible in the image
Color consistency: HSV thresholds may need adjustment under different lighting conditions
File naming: Ensure filenames contain a valid well identifier (e.g., A01, H06)

##  License
MIT License
