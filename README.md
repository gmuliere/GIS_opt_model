# HeatNODE
This project consists of a GIS-based optimization model to analyze the potential of district heating on a local scale, exploiting renewable heat resources and waste heat, in Italy.
## Prerequisites
- Python 3.9 or later
- Optimization solver such as CBC or GLPK
##  Used Libraries
Python libraries employed:
- geopandas (v 0.14.0)
- pandas (v  2.1.0)
- numpy (v 1.25.2)
- osmnx (v 1.8.1)
- oemof.solph (v 0.5.2)
## Installation
Download or clone the Model folder from the GitHub link.
## Usage
- Compile the _data.xlsx_ file with your yearly data.
- Run the _model.py_ file.
### notes
1. A pre-compiled version of the input file, _data.xlsx_, is provided as an example and for testing the model. The dataset is the result of analyses conducted by Politecnico di Milano.
2. The input folder contains a shapefile (_Com01012021_WGS84.shp_) with geographical information about Italian municipalities. The source of this file is the Italian Institute of Statistics (ISTAT).
