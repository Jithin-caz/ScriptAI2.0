# Handwritten Text Recognition Pipeline: Automated Text Extraction from Handwritten Documents

The HTR Pipeline is a comprehensive solution for extracting text from handwritten documents using advanced computer vision and machine learning techniques. This pipeline combines word detection, line clustering, and text recognition to accurately transcribe handwritten content into digital text.

The system employs a multi-stage approach that first detects individual words using bounding boxes, clusters them into lines, and then performs text recognition using either a best-path decoder or a word beam search algorithm with dictionary support. The pipeline is particularly effective for processing both single-line and multi-line handwritten text, with configurable parameters to handle different writing styles and document layouts.

## Repository Structure
```
htr_pipeline/                 # Main package directory containing core functionality
├── models/                   # Pre-trained models and character list files
├── reader/                   # Text recognition implementation using CTC decoding
│   ├── __init__.py          # Reader initialization and main functions
│   └── ctc.py               # CTC decoder implementation
└── word_detector/           # Word detection and line clustering components
    ├── aabb.py              # Axis-aligned bounding box implementation
    ├── aabb_clustering.py   # Clustering algorithm for word grouping
    ├── coding.py           # Encoding utilities for detection
    └── iou.py              # Intersection over Union calculations
scripts/                     # Application entry points and demos
├── demo.py                 # Command-line demo script
└── streamlit_run.py        # Interactive web interface using Streamlit
data/                       # Configuration and sample data
└── config.json            # Sample configurations and parameters
```

## Usage Instructions
### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for improved performance)

Required Python packages:
- numpy
- onnxruntime
- opencv-python
- scikit-learn
- editdistance
- path
- streamlit (for web interface)

### Installation
```bash
# Clone the repository
git clone https://github.com/githubharald/HTRPipeline.git
cd HTRPipeline

# Create and activate virtual environment
## Windows
python -m venv py39_env
py39_env\Scripts\activate

## macOS/Linux
python3 -m venv py39_env
source py39_env/bin/activate

# Install the package
pip install -e .
```

### Quick Start
```python
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig
import cv2

# Load image
img = cv2.imread('path/to/image.png', cv2.IMREAD_GRAYSCALE)

# Configure and run pipeline
read_lines = read_page(
    img,
    detector_config=DetectorConfig(scale=1.0, margin=1),
    line_clustering_config=LineClusteringConfig(min_words_per_line=2),
    reader_config=ReaderConfig(decoder='best_path')
)

# Print results
for line in read_lines:
    print(' '.join(word.text for word in line))
```

### More Detailed Examples
1. Using dictionary-based word beam search:
```python
from htr_pipeline import PrefixTree

# Load dictionary
with open('data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

# Configure reader with dictionary
read_lines = read_page(
    img,
    detector_config=DetectorConfig(scale=1.0, margin=1),
    line_clustering_config=LineClusteringConfig(min_words_per_line=2),
    reader_config=ReaderConfig(decoder='word_beam_search', prefix_tree=prefix_tree)
)
```

### Troubleshooting
Common issues and solutions:

1. Word Detection Issues
- Problem: Words not properly detected
- Solution: Adjust the scale parameter in DetectorConfig
```python
detector_config = DetectorConfig(scale=0.5)  # Try different scale values
```

2. Line Clustering Problems
- Problem: Words incorrectly grouped into lines
- Solution: Modify min_words_per_line parameter
```python
line_config = LineClusteringConfig(min_words_per_line=3)
```

3. Recognition Accuracy
- Problem: Poor text recognition results
- Solution: Enable dictionary-based decoding and ensure proper image preprocessing
```python
# Ensure proper image preprocessing
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
```

## Data Flow
The pipeline processes images through three main stages: word detection, line clustering, and text recognition, transforming raw handwritten text images into digitized text output.

```ascii
Input Image → Word Detection → Line Clustering → Text Recognition → Output Text
     ↓              ↓               ↓                   ↓
[Raw Image] → [Bounding Boxes] → [Line Groups] → [Recognized Words]
```

Component interactions:
1. Word Detector processes input image to locate individual words using bounding boxes
2. Line Clustering groups detected words into lines using DBSCAN algorithm
3. Text Recognition applies CTC decoding to extract text from word images
4. Dictionary support provides word beam search for improved accuracy
5. Scale and margin parameters control detection sensitivity
6. Minimum words per line parameter affects line grouping
7. Output includes both recognized text and visualization options