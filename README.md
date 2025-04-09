# MindStream

An interactive Streamlit application for exploratory data analysis (EDA) and machine learning model creation, all within a user-friendly web interface.

## Overview

MindStream empowers users to perform complete data analysis and machine learning workflows without writing code. From data upload to visualization, preprocessing, model training, and prediction—all through an intuitive browser interface.

## Features

- **Data Upload**: Upload CSV files with custom delimiter options
- **Exploratory Data Analysis (EDA)**:
  - View summary statistics
  - Visualize distributions of numeric features
  - Generate bar plots for categorical features
  - Create correlation heatmaps
- **Data Preprocessing**:
  - Handle missing values with multiple strategies
  - Automatic encoding of categorical variables
  - Train-test splitting
- **Machine Learning**:
  - Support for classification and regression problems
  - Multiple algorithms (Random Forest, SVM, KNN, AdaBoost)
  - Performance metrics visualization
- **Prediction Interface**:
  - Make predictions with trained models
  - Interactive input form for new data points

## Screenshot

*Add a screenshot of your application here*

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mindstream.git
   cd mindstream
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is missing, install the main dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the application** in your browser at `http://localhost:8501`

## Usage Guide

### 1. Upload Tab
- Upload your CSV dataset
- Select the appropriate delimiter for your file

### 2. EDA Tab
- Review summary statistics of your data
- Explore data visualizations
- Handle missing values using your preferred strategy

### 3. ML Tab
- Select your target variable
- Choose between classification and regression
- Train multiple models and compare their performance

### 4. Prediction Tab
- Select a trained model
- Input values for each feature
- Get predictions for new data points

## Supported Machine Learning Models

### Classification
- Random Forest Classifier
- Support Vector Machine
- K-Nearest Neighbors
- AdaBoost Classifier

### Regression
- Random Forest Regressor
- Support Vector Regressor
- K-Nearest Neighbors Regressor
- AdaBoost Regressor

## Technical Details

MindStream is built using:
- **Streamlit**: For the interactive web interface
- **Pandas**: For data manipulation and analysis
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For machine learning algorithms and preprocessing

## Future Enhancements

- Add support for more file formats (Excel, JSON, etc.)
- Implement feature importance analysis
- Add more machine learning algorithms
- Include hyperparameter tuning capabilities
- Enable model export and saving
- Add time series analysis capabilities
- Implement cross-validation options

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for making data apps easy to create
- [Scikit-learn](https://scikit-learn.org/) for their excellent machine learning library
- [Pandas](https://pandas.pydata.org/) for powerful data analysis tools
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization capabilities

---

Created with ❤️ by Koceila djaballah

*Last updated: April 2025*
