# Pharmaceutical Sales Analytics Dashboard

Enterprise-grade analytics platform for pharmaceutical market intelligence.

## Features

- **🌍 Geographic Analysis**: Interactive world map with regional insights
- **📈 Market Intelligence**: Manufacturer ranking and molecule analysis
- **💰 Price Analytics**: Price positioning and trend analysis
- **📊 Advanced Insights**: Market concentration metrics and Pareto analysis
- **🔬 Portfolio Analysis**: Manufacturer portfolio efficiency

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Data Format

Upload Excel or CSV files with the following structure:

- Required columns: `Manufacturer`, `Molecule`, `Country`, `Region`
- MAT Period columns: `MAT Q4 2023 USD MNF`, `MAT Q4 2023 Standard Units`, etc.
- Optional columns: Various metadata columns

## Deployment

The app is configured for deployment on Streamlit Cloud. Simply connect your GitHub repository.

## License

Proprietary - For internal use only