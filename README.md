# Automated Invoice Data Extraction

This repository contains the code and data for an automated invoice data extraction project using computer vision and deep learning techniques.

## Project Structure
- `dataset`: Contains the dataset used for the project.
- `notebook`: Jupyter notebook for data analysis and model training.
- `requirements`: List of dependencies required to run the project.

## Getting Started

### Prerequisites
- Python 3.8+
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/muhammadahmedzaheer/Invoice-Data-Extraction.git
   cd Automated-Invoice-Data-Extraction

2. Install the required packages:
   ```
   pip install -r requirements.txt

## Usage

1. Set up the project environment:
   ```
   import torch
   from IPython.display import Image, clear_output
   from utils.downloads import attempt_download

   # Clear any previous output
   clear_output()
   print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

2. Download the dataset:
   ```
   from roboflow import Roboflow
   rf = Roboflow(api_key="your_api_key")
   project = rf.workspace("test-t1s3e").project("invoice-5wfdh")
   version = project.version(1)
   dataset = version.download("yolov5")

3. Train the YOLOv5 model:
   ```
   import yaml
   with open(dataset.location + "/data.yaml", 'r') as stream:
       num_classes = str(yaml.safe_load(stream)['nc'])

   from utils.plots import plot_results
   Image(filename='/kaggle/working/yolov5/runs/train/yolov5s_results/results.png', width=1000)  # view results.png

4. Display results:
   ```
   import glob
   from IPython.display import display, Image

   thumbnail_width = 900
   thumbnail_height = 900
   for imageName in glob.glob('/kaggle/working/yolov5/runs/train/yolov5s_results/*.png')[:100]:
       display(Image(filename=imageName, width=thumbnail_width, height=thumbnail_height))

5. Perform OCR on the detected invoice sections:
   ```
   import os
   import pytesseract
   from PIL import Image

   detect_folder = '/kaggle/working/yolov5/runs/detect/exp5'
   crops_folder = os.path.join(detect_folder, 'crops')

   count = 0
   for image_file in os.listdir(detect_folder):
       if image_file.endswith('.jpg') or image_file.endswith('.png'):
           image_path = os.path.join(detect_folder, image_file)
           image_name = os.path.splitext(image_file)[0]
           print("_____________________________________________________________")
           print(image_name)
           for class_folder in os.listdir(crops_folder):
               class_path = os.path.join(crops_folder, class_folder)
               if os.path.isdir(class_path):
                   class_text = []
                   for crop_file in os.listdir(class_path):
                       if os.path.splitext(crop_file)[0] == image_name:
                           if crop_file.endswith('.jpg') or crop_file.endswith('.png'):
                               crop_image_path = os.path.join(class_path, crop_file)
                               try:
                                   crop_text = pytesseract.image_to_string(Image.open(crop_image_path))
                                   print(class_folder)
                                   print(crop_text.strip())
                                   print('-' * 30)
                               except Exception as e:
                                   print(f"An error occurred during OCR for {crop_image_path}: {e}")
                               break
       count += 1
       if count > 20:
           break

## Acknowledgements
1. YOLOv5
2. Roboflow
3. Pytesseract
