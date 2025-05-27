# COVID-19 Chest X-ray Federated Learning Project

This repository contains all the scripts needed for the CXR classification use-case.

## Project Structure

```
/mnt/d/irina/paroma/scripts/datasets/chest_xray_covid/
├── src/
│   ├── features_viz.ipynb
│   ├── pandas_splitter.ipynb
│   ├── plot_generator.ipynb
│   ├── scores.ipynb
│   ├── fl_splitter.ipynb
│   └── utils/
├── datasets/
├── models/
└── results/
```

## Dataset

The project uses the **COVID-19 Radiography Dataset** with 4 disease classes:
- COVID-19
- Lung Opacity
- Normal
- Viral Pneumonia

## Key Components

### Data Preparation & Splitting

#### `pandas_splitter.ipynb`
**Purpose**: Dataset preparation and CSV file generation
- Converts image directory structure to CSV format for easier data handling
- Creates train/validation/test splits from the COVID-19 radiography dataset
- Generates dummy/one-hot encoded labels for multi-class classification
- Handles path preprocessing for different dataset configurations

#### `fl_splitter.ipynb`
**Purpose**: Federated learning data partitioning
- Implements sophisticated data splitting strategies for federated scenarios
- Supports both balanced (IID) and unbalanced (non-IID) data distributions
- Uses Dirichlet distribution for realistic federated data heterogeneity
- Handles missing class scenarios (some clients don't have all disease types)
- Creates client-specific datasets for federated learning experiments

### Model Analysis & Visualization

#### `features_viz.ipynb` (EXPERIMENTAL)
**Purpose**: Feature extraction and visualization for model analysis
- Loads trained ResNet18 models and extracts deep features from test data
- Performs dimensionality reduction using PCA and UMAP
- Creates 2D/3D visualizations of learned representations
- Saves features and embeddings
- Includes custom dataset class for loading COVID-19 radiography data

#### `plot_generator.ipynb`
**Purpose**: Visualization and analysis tool
- Generates data distribution plots showing class imbalance across federated clients
- Creates training curve visualizations (loss/accuracy evolution)
- Provides statistical analysis tools (accuracy, precision, recall, F1-score)
- Supports comparison between centralized and federated learning results
- Includes styling functions for publication-quality plots

### Model Evaluation

#### `scores.ipynb`
**Purpose**: Model evaluation and performance analysis
- Loads trained models (PyTorch and ONNX formats) for inference
- Generates predictions on test datasets
- Creates confusion matrices with normalization
- Computes classification metrics (accuracy, sensitivity, precision, F1-score)
- Includes error analysis and misclassification visualization
- Supports both centralized and federated model evaluation

## Technical Features

### Model Architecture
- **ResNet18** as the primary backbone (modified for single-channel input)
- Lightning PyTorch framework for training organization
- ONNX export capability for deployment and inference

### Federated Learning Capabilities
- **Data heterogeneity simulation** through various splitting strategies
- **Missing class scenarios** where clients have different disease types
- **Performance comparison** between centralized and federated approaches
- **IID vs Non-IID** data distribution analysis

### Analysis Tools
- **Feature space analysis** using PCA/UMAP embeddings
- **Training dynamics** visualization with loss/accuracy curves
- **Performance metrics** with statistical significance testing
- **Data distribution** analysis across federated clients

## Dataset Handling

### Custom PyTorch Components
- Custom Dataset classes for loading grayscale chest X-rays (224x224 resolution)
- Support for various image transformations and preprocessing
- One-hot encoded labels for multi-class classification
- Flexible data loading for different federated scenarios

### Data Distribution Strategies
1. **Balanced (IID)**: Equal distribution of all classes across clients
2. **Unbalanced (Non-IID)**: Realistic federated scenarios with data heterogeneity
3. **Missing Classes**: Some clients don't have access to all disease types
4. **Dirichlet Distribution**: Statistical approach to create realistic data splits

## Usage

Here is the workflow of the repo:

1. **Data Preparation**: Use `pandas_splitter.ipynb` and `fl_splitter.ipynb` to prepare datasets
2. **Model Training**: Train ResNet18 models on federated data splits
3. **Visualization**: Use `features_viz.ipynb` for feature analysis and `plot_generator.ipynb` for visualizations
4. **Evaluation**: Use `scores.ipynb` for model evaluation

## Research Applications

This framework is designed for research in:
- Federated learning for medical imaging
- Non-IID data distribution effects
- Privacy-preserving machine learning in healthcare
- COVID-19 diagnosis from chest X-rays
- Comparative analysis of centralized vs. federated approaches

## Dependencies

- PyTorch & Lightning PyTorch
- scikit-learn
- pandas & numpy
- matplotlib & seaborn
- ONNX & ONNXRuntime
- PIL (Pillow)
- plotly (for 3D visualizations)