import os
import pandas as pd
import numpy as np
import sys
import tempfile
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
import ipywidgets as widgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QComboBox
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
#path for whereever df_final.csv is stored
path = "C:\\Users\\uygar\\Desktop\\DataAnalytics\\export_40"
os.chdir(path)


##Logging
def redirect_print_to_logfile(logfile):
    # Remove the existing logfile if it exists
    if os.path.exists(logfile):
        os.remove(logfile)
    
    # Open the logfile in append mode
    sys.stdout = open(logfile, "a")
# Set the logfile name
logfile = "output.log"
# Redirect print statements to the logfile
redirect_print_to_logfile(logfile)



##Heat Map
class Location:
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        self.db_values = dict()



def load_df():
    df = pd.read_csv("df_final.csv")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return df

def get_locations(df, noise_type):
    locations = []
    for _, row in df.iterrows():
        locations.append(Location(row['date'], row['latitude'], row['longitude'], row[noise_type]))
    return locations

def init_map():
    map = folium.Map(location=[locations[0].lat, locations[0].long], tiles='CartoDB Positron', zoom_start=15, zoom_control=False)
    return map

def update_map(noise_type):
    print(f"Updating map with noise type: {noise_type}") #debug see if function execs

    # Generate all data points
    all_data = {}
    for loc in locations:
        for date, noise_data in loc.db_values.items():
            if date not in all_data:
                all_data[date] = []
            
            data_value = loc.db_values[date][noise_type]
            print(f"Data value: {data_value}")  # debug, data values
            all_data[date].append([loc.lat, loc.long, data_value])

    # Convert the dictionary to a list sorted by time
    heatmap_data_time = [all_data[date] for date in sorted(all_data.keys())]
    #Bias for showing 0s as blue?
    #heatmap_data_time = [[[lat, lon, value + 1] for [lat, lon, value] in date_data] for date_data in heatmap_data_time]

    global map
    map = init_map()
    # add HeatMapWithTime instead of regular HeatMap
    gradient = {0: 'blue', 0.22:'purple',0.33: 'yellow', 0.77: 'orange', 1: 'red'}
    heatmap_layer = HeatMapWithTime(heatmap_data_time, auto_play=True, max_opacity=0.8, gradient=gradient,radius=30)
    heatmap_layer.add_to(map)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html",dir=path)
    temp_file.close()
    map.save(temp_file.name)

    web_view.load(QUrl.fromLocalFile(os.path.abspath(temp_file.name)))
    web_view.reload()

def remove_temp_file():
    try:
        os.unlink(temp_file.name)
    except:
        pass

def on_noise_type_changed(noise_type):
    print(f"Changing noise type to {noise_type}") #debug see if function execs
    update_map(noise_type)
 
# Load the dataframe
df = load_df()
# Create list of unique location tuples
location_tuples = df[['latitude', 'longitude']].drop_duplicates().values

# Create list of Location objects
locations = [Location(lat, long) for lat, long in location_tuples]

# Iterate over dataframe rows and update db_values for each location
for index, row in df.iterrows():
    # Find location in locations list
    for location in locations:
        if location.lat == row['latitude'] and location.long == row['longitude']:
            # Add db_values to this location
            location.db_values[row['date']] = {
                'Human voice - Shouting': row['Human voice - Shouting'],
                'Human voice - Singing': row['Human voice - Singing'],
                'Music non-amplified': row['Music non-amplified'],
                'Nature elements - Wind': row['Nature elements - Wind'],
                'Transport road - Passenger car': row['Transport road - Passenger car'],
                'Transport road - Siren': row['Transport road - Siren'],
                'Unsupported': row['Unsupported']
            }


# Get the list of noise types
noise_types = df.columns.drop(['date', 'latitude', 'longitude','Unnamed: 0', 'description'])

# Initialize the application
app = QApplication(sys.argv)
window = QWidget()
layout = QVBoxLayout()

# Create a combo box for noise type selection
combo_box = QComboBox()
combo_box.addItems(noise_types)
combo_box.currentTextChanged.connect(on_noise_type_changed)
layout.addWidget(combo_box)

# Initialize the web view
web_view = QWebEngineView()
web_view.loadFinished.connect(remove_temp_file)
layout.addWidget(web_view)

window.setLayout(layout)
window.show()

# Update map with the first noise type
update_map(noise_types[0])

# Run the application
sys.exit(app.exec_())        
