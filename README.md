# Agro scripting

## Background

This script developed for the department of Agriculture of the Hebrew
University of Jerusalem, By Ofer Ezrachi.

## Installation

Make sure you installed the python interpreter (3.7 or above), as well as all
the listed libraries in the requirements.txt file.
<br>
Note: if you are using JetBrains Pycharm as your IDE, you can just open the
requirements.txt and in the top section the IDE will ask you if you want to
install the listed libraries.

## Data preparations

### Plait

Make sure to supply to the script a plait similar to the plaits in the data
folders. The plait should be in csv format.

### Input

Make sure to save the sampling machine output as csv in UTF-8 format, otherwise
the script won't be able to load the file. Also, make sure you delete any
degrees symbol.

### Set parameters

In the top of the script there are some parametrs you need to set:

- PLAIT_PATH= The plait file path. Example='data/plait2.csv'
- INPUT_PATH = The input file path. Example='data/input.csv'
- OUTPUT_PATH = The output folder path. Example='output/'
- DATA_COL_START = The column the input starts from to start in input. Example
  = 3
- REMOVEABLE_CHARS = Chars to remove in plait mapping. Example = ['%']
- LUM_START = Row that the lum section start. Example = 101
- OD_START = Row that the od section. Example= 48 start
- SAMPLES = Amount of samples in the input at each section. Example =  49 
- GRAPH_NAME = Title of graph for each graph. Example= {'od': 'OD', 'lum': 'LUM',
  'lum_div_od': 'LUM/OD'}  
- GRAPH_YLABEL = Y label of each graph. Example = {'od': 'OD', 'lum': 'LUM (RLU)',
  'lum_div_od': 'LUM/OD'} # 

### Run
Just run the script. Enjoy :)

## Support
Feel free to reach out for help- [oferezr@gmail.com](oferezr@gmail.com)