# Pharmaceutical Market Intelligence Platform

## 📋 Overview

Advanced analytics dashboard for pharmaceutical MAT (Moving Annual Total) data. This Streamlit application provides comprehensive market analysis, trends, and insights for pharmaceutical sales data.

## ✨ Features

- **📊 Executive Dashboard**: Key metrics and performance indicators
- **🏭 Manufacturer Analysis**: Market share, rankings, and growth analysis
- **🧪 Molecule Analysis**: Product-level insights and trends
- **📈 Trend Analysis**: Historical performance tracking
- **🌍 Geographic Analysis**: Country-level market insights
- **💰 Pricing Analysis**: Price distribution and trends
- **📋 Data Explorer**: Interactive data viewing and export

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Step 1: Clone or Download

Download the application files:
- `pharma_analytics_app.py`
- `requirements.txt`

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Running the Application

### Start the Application

```bash
streamlit run pharma_analytics_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Alternative: Specify Port

```bash
streamlit run pharma_analytics_app.py --server.port 8080
```

## 📁 Data Format

### Required Columns

Your data file should contain:

- `Source.Name` - Data source identifier
- `Country` - Market country
- `Manufacturer` - Company name
- `Molecule` - Active ingredient
- `Molecule List` - List of molecules

### MAT Period Columns

The application automatically detects columns in these formats:

```
MAT Q3 2022 USD MNF
MAT Q3 2022 Standard Units
MAT Q3 2022 Units
MAT Q3 2022 SU Avg Price USD MNF
MAT Q3 2022 Unit Avg Price USD MNF
```

### Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

### Example Data Structure

```csv
Manufacturer,Molecule,Country,MAT Q3 2022 USD MNF,MAT Q3 2022 Standard Units,...
ABBOTT,PENICILLIN G,ALGERIA,2265,7065,...
ASPEN,NADROPARIN CALCIUM,ALGERIA,13,10,...
```

## 🎨 Features Guide

### 1. Dashboard Tab
- View key performance metrics
- Market share visualization
- Growth analysis charts

### 2. Manufacturers Tab
- Top 20 manufacturers ranking
- Market share percentages
- Growth rates

### 3. Molecules Tab
- Product performance analysis
- Market share by molecule
- Visual rankings

### 4. Trends Tab
- Historical performance tracking
- Period-over-period analysis
- Top performer trends

### 5. Geography Tab
- Country-level analysis
- Geographic market share
- Regional performance

### 6. Pricing Tab
- Price distribution analysis
- Manufacturer pricing comparison
- Average price trends

### 7. Data Tab
- Full data exploration
- Column statistics
- Missing value analysis
- Data export options

## 🔧 Configuration

### Sidebar Controls

- **MAT Period Selector**: Choose analysis period
- **Metric Selector**: Select between Sales, Volume, or Price
- **Manufacturer Filter**: Filter by specific manufacturers
- **Molecule Filter**: Filter by specific molecules

### Export Options

- **CSV Export**: Download processed data as CSV
- **Excel Export**: Download with formatted sheets

## 🐛 Troubleshooting

### Common Issues

**Issue**: Module not found error
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --upgrade
```

**Issue**: Port already in use
```bash
# Solution: Use different port
streamlit run pharma_analytics_app.py --server.port 8502
```

**Issue**: Data not loading
- Ensure file format is CSV or Excel
- Check that MAT columns follow the expected naming pattern
- Verify column headers are in the first row

### Data Processing Issues

If data doesn't process correctly:
1. Check column names match expected formats
2. Ensure numeric columns don't contain text
3. Verify MAT period columns follow pattern: "MAT Q[1-4] [YYYY]"

## 📊 Performance Tips

### For Large Datasets

1. **Filter Early**: Use sidebar filters to reduce data volume
2. **Select Specific Period**: Analyze one period at a time
3. **Export Subsets**: Export filtered data for detailed analysis

### Memory Optimization

- Close unused browser tabs
- Clear session and restart if app becomes slow
- Process data in batches if file is very large

## 🔐 Security Notes

- This application processes data locally
- No data is sent to external servers
- Confidential data remains on your machine

## 📝 Version History

### Version 3.1.0
- Initial release
- Core analytics features
- Multi-tab dashboard
- Export functionality
- Interactive visualizations

## 🆘 Support

### Error Messages

The application provides detailed error messages. Common solutions:

- **"Unsupported file format"**: Use CSV or Excel files only
- **"No MAT period columns found"**: Check column naming format
- **"Error processing data"**: Verify data structure and formats

### Getting Help

1. Check this README for solutions
2. Verify data format matches examples
3. Review error messages in the application

## 📄 License

Confidential - For Internal Use Only

## 👥 Credits

Developed for pharmaceutical market intelligence and analysis.

---

**Note**: This application is designed for MAT (Moving Annual Total) pharmaceutical data analysis. Ensure your data follows the expected format for optimal results.
