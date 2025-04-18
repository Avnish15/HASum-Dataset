# Video Summarization and Annotation Pipeline

This repository provides a comprehensive pipeline for video summarization, annotation, and downloading videos from YouTube for training video summarization models. The pipeline allows the processing and annotation of video shots, application of manual summaries, generation of summary scores, and storage of annotations in both Excel and HDF5 formats. The pipeline also includes functionality to download videos from YouTube based on ActivityNet annotations.

## Features

- **Video Annotation**: Annotate video shots manually and apply a summary. Store the results in an HDF5 file and an Excel file for future use.
- **HDF5 Visualization**: Allows visualization and comparison of video annotations and summaries within HDF5 files.
- **YouTube Video Downloading**: Downloads videos from YouTube using `yt-dlp` based on ActivityNet categories, filtering by video duration (≥3 minutes) and resolution (720p or 480p).
- **ActivityNet Integration**: Filter and download videos from ActivityNet based on predefined categories, such as 'Baking cookies', 'Rock climbing', and others. Download video clips that match the required metadata, ensuring compatibility for summarization tasks.

## Installation

To get started, you need to install the required dependencies. You can install them using the following command:

```bash
pip install opencv-python h5py pandas yt-dlp tqdm
```

## Usage

### 1. **Video Annotation**

You can annotate videos by running the `aggregator.py` script. The script allows you to annotate video shots and apply manual summaries. The annotations are stored in both HDF5 and Excel formats.

```bash
python aggregator.py --input_dir <input_directory> --output_dir <output_directory> --sample_rate <sample_rate>
```

- `input_dir`: Path to the directory containing your video files.
- `output_dir`: Path to the directory where the annotated HDF5 and Excel files will be stored.
- `sample_rate`: Specify the sampling rate for video frame extraction.

### 2. **HDF5 Visualization**

To visualize and check the contents of an HDF5 file, you can use the `data_visualization.py` script. This allows you to view the annotations and summaries stored in the HDF5 file.

```bash
python data_visualization.py <path_to_h5_file>
```

- `path_to_h5_file`: Path to the HDF5 file you wish to visualize.

### 3. **Video Downloading**

The repository also includes a script for downloading videos from YouTube based on the categories in the ActivityNet dataset. To download videos, run the following:

```bash
python main.py
```

This script downloads videos from YouTube, ensuring that only videos with a duration of at least 3 minutes and a resolution of 720p or 480p (preferably 720p) are selected.

### 4. **Configuration**

Adjust parameters for video downloading, annotation, and metadata filtering by modifying the respective scripts. You can set:

- Duration and resolution preferences for video downloads.
- Sampling rate and frame extraction parameters for video annotation.
- Video categories for downloading based on ActivityNet metadata.

### 5. **Video Summarization**

For video summarization tasks, you can use the annotated data from the HDF5 and Excel files for training models such as VASNet, PGL-SUM, and Hierarchical Transformer. This repository includes pre-configured formats for training and evaluating summarization models on the annotated videos.

## Example Workflow

1. **Download Videos**: Use `main.py` to download a set of videos from ActivityNet based on chosen categories.
2. **Annotate Videos**: Use `aggregator.py` to annotate the downloaded videos by applying a manual or model-generated summary.
3. **Store Annotations**: Annotations are stored in both HDF5 files and Excel format for easy future use and evaluation.
4. **Train Summarization Models**: Use the stored annotations for training video summarization models (e.g., VASNet, PGL-SUM).
5. **Visualize and Compare Annotations**: Use `data_visualization.py` to compare ground truth and model summaries for evaluation.


## Acknowledgments

- **ActivityNet**: Used for providing the video categories and metadata.
- **yt-dlp**: Utilized for downloading videos from YouTube.
- **OpenCV**: Used for video processing and frame extraction.
- **H5py**: Used for handling HDF5 files to store video annotations and summaries.

---

This pipeline serves as a valuable tool for video summarization tasks, enabling video download, annotation, and storage for future summarization model training.
