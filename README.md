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
   ```bash
   git clone https://github.com/muhammadahmedzaheer/Automated-Invoice-Data-Extraction.git
   cd Automated-Invoice-Data-Extraction

2. Install the required packages:
   ```bash
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
