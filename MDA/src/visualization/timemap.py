import os 
import pandas as pd
import numpy as np
import sys
import tempfile
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
import ipywidgets as widgets
import tkinter as tk
from tk_html_widgets import HTMLLabel
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl

path = "C:\\Users\\uygar\\Desktop\\DataAnalytics\\export_40"
os.chdir(path)
# %%
class Location:
    def __init__(self, identifier, lat, long):
        self.identifier = identifier
        self.lat = lat
        self.long = long
        self.row = 1
    def get_address(self):
        parts = self.identifier.split("_")
        if len(parts) < 5:
            return "Invalid identifier"
        street_name = parts[4].replace("-", " ").replace(".csv", "")
        return street_name + ", Leuven, Belgium"
    
    def get_all_db_values(self):
        filename = os.path.splitext(self.identifier)[0] + ".csv"
        if not os.path.isfile(filename):
            print(f"File does not exist: {filename}")  #debug and error check
            return []
        df = pd.read_csv(filename, sep=";", engine="python")
        df['result_timestamp'] = pd.to_datetime(df['result_timestamp'], format='%d/%m/%Y %H:%M:%S.%f')
        db_values = df["laf005_per_hour"]
        time_values = df['result_timestamp']
        #print(time_values.head())  # debug
        #print(db_values.head())  #debug
        return list(zip(db_values, time_values))



# %%
# Create Location objects for each location
locations = [
    Location("csv_results_40_255439_mp-01-naamsestraat-35-maxim.csv", 50.877160828051615, 4.7006992525313445),
    Location("csv_results_40_255440_mp-02-naamsestraat-57-xior.csv", 50.87649057167669, 4.700691639839767),
    Location("csv_results_40_255441_mp-03-naamsestraat-62-taste.csv", 50.875850541602986, 4.700192168675618),
    Location("csv_results_40_303910_mp-04-his-hears.csv", 50.87525620749695, 4.700106135276407),
    Location("csv_results_40_255442_mp-05-calvariekapel-ku-leuven.csv", 50.87448678827803, 4.699889625684066),
    Location("csv_results_40_255443_mp-06-parkstraat-2-la-filosovia.csv", 50.87409073433792, 4.700018371714718),
    Location("csv_results_40_255444_mp-07-naamsestraat-81.csv", 50.87381822269834, 4.70010170607578),
    Location("csv_results_40_255445_mp-08-kiosk-stadspark.csv", 50.87526203165543, 4.701475911003796),
    Location("csv_results_40_280324_mp08bis---vrijthof.csv", 50.87890579999999, 4.7011444846213415),
]


# %%
def init_map():
    map = folium.Map(location=[locations[0].lat, locations[0].long], tiles='CartoDB Positron', zoom_start=15, zoom_control=False)
    return map


# %%


# %%
def update_map():

    # Generate all data points
    all_data = {}
    for loc in locations:
        db_values = loc.get_all_db_values()
        for db_value, time_value in db_values:
            if time_value not in all_data:
                all_data[time_value] = []
            all_data[time_value].append([loc.lat, loc.long, db_value])
    #print(all_data.keys()) #debug
    # Convert the dictionary to a list sorted by time
    heatmap_data_time = [all_data[time] for time in sorted(all_data.keys())]

    global map
    map = init_map()
    # add HeatMapWithTime instead of regular HeatMap
    heatmap_layer = HeatMapWithTime(heatmap_data_time, auto_play=True, max_opacity=0.8)
    heatmap_layer.add_to(map)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    temp_file.close()
    map.save(temp_file.name)

    web_view.load(QUrl.fromLocalFile(os.path.abspath(temp_file.name)))


def remove_temp_file():
    try:
        os.unlink(temp_file.name)
    except:
        pass

#This is to show the map in a separate window
app = QApplication(sys.argv)
window = QWidget()
layout = QVBoxLayout()


web_view = QWebEngineView()
web_view.loadFinished.connect(remove_temp_file)
layout.addWidget(web_view)

window.setLayout(layout)
window.show()

#Update map with time info
update_map()

sys.exit(app.exec_())



