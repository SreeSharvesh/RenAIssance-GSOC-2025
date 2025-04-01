import subprocess
import os

book_name = "Ezcaray - Vozes"
# Define the paths and parameters
book_path = f"/home/sharvesh/Documents/Others/Human_AI/Test_sources/Test sources/{book_name}.pdf"
output_dir = f"/home/sharvesh/Documents/Others/Human_AI/model/processed_book/{book_name}/pages"
trained_model = "/home/sharvesh/Documents/Others/Human_AI/model/CRAFT/weights/craft_mlt_25k.pth"
refiner_model = "/home/sharvesh/Documents/Others/Human_AI/model/CRAFT/weights/craft_refiner_CTW1500.pth"
text_threshold = 0.9
test_folder = output_dir  # Images folder created by process_main_utils.py
result_folder = "/home/sharvesh/Documents/Others/Human_AI/model/result"
contour_line_splitter_script = "/home/sharvesh/Documents/Others/Human_AI/model/contour_line_splitter.py"
line_segments_folder = f"/home/sharvesh/Documents/Others/Human_AI/model/processed_book/{book_name}/line_segments"
padding = 50 #fixed(50)
min_width = 50
threshold = 0.7
margin = 0

# New parameters for process_main_utils.py
dpi = 300
remove_borders = True
noise_removal_area_threshold = 50
intensity_threshold = 128

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
os.makedirs(line_segments_folder, exist_ok=True)

# Check if the output directory is empty
if not os.listdir(output_dir):
    print("Output directory is empty. Performing Step 1: Processing PDF into individual images.")
    # Step 1: Process the PDF into individual images
    subprocess.run([
        "python3", "process_main_utils.py",
        book_path, output_dir,
        "--dpi", str(dpi),
        "--remove_borders",
        "--noise_removal_area_threshold", str(noise_removal_area_threshold),
        "--intensity_threshold", str(intensity_threshold)
    ])
else:
    print("Output directory is not empty. Skipping Step 1.")

# Step 2: Run the text detection model on the processed images
print("Performing Step 2: Running the text detection model.")
subprocess.run([
    "python3", "/home/sharvesh/Documents/Others/Human_AI/model/CRAFT/test.py",
    "--trained_model", trained_model,
    "--text_threshold", str(text_threshold),
    "--test_folder", output_dir,
    "--refine",
    "--refiner_model", refiner_model
])

# Step 3: Run the contour line splitter
print("Performing Step 3: Running the contour line splitter.")
subprocess.run([
    "python3", "contour_line_splitter.py",
    output_dir,  # Image directory from the first process
    result_folder,  # Contour directory from the second process
    line_segments_folder,  # Output directory for line segments
    "--padding", str(padding),
    "--min_width", str(min_width),
    "--threshold", str(threshold),
    "--margin", str(margin),
    "--visualize"
])

print("Processing completed successfully.")
