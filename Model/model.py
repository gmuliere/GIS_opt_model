

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import folium
import pyproj
import os
import logging
# import oemof.solph as solph
# import chardet
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry import LineString
from scipy.spatial import  Delaunay
from geopy.distance import geodesic
from matplotlib.patches import Polygon
from scipy.spatial import cKDTree
from folium.plugins import MarkerCluster, LocateControl
# from oemof.tools import logger
# from shapely.ops import transform
# import contextily as ctx
# from functools import partial


input_path = os.path.abspath("input")
output_path = os.path.abspath("output")

################################################## GEO DATAFRAME WITH HEAT DEMANDS #############################################
input_filename = "data_02.xlsx"
input_filepath = os.path.join(input_path, input_filename)
df_1 = pd.read_excel(input_filepath, sheet_name = 'dmd') #, encoding='latin1')
geometry = gpd.points_from_xy(df_1["longitude"], df_1["latitude"])
geo_df_dmd = gpd.GeoDataFrame(df_1, geometry=geometry)
# print(geo_df.head())

######################## heat demands - shp file ##############################
# output_filename = "heat_demands.shp"
# output_filepath = os.path.join(output_path, output_filename)
# geo_df_dmd.to_file(output_filepath, encoding= 'utf-8')

######################################################### GEO DATAFRAME WITH GEO - SOURCES ################################

df_geo = pd.read_excel(input_filepath, sheet_name='GEO')
geo_sources = df_geo[(df_geo['likely_geo'] == 1) | (df_geo['geo_1000_2000'] > 0)]
geometry_geo_sources = gpd.points_from_xy(geo_sources['longitude'], geo_sources['latitude'])
geo_df_geo_sources = gpd.GeoDataFrame(geo_sources, geometry=geometry_geo_sources, crs='EPSG:4326')
geometry_geo_sources = gpd.points_from_xy(geo_sources['longitude'], geo_sources['latitude'])
geo_df_geo_sources = gpd.GeoDataFrame(geo_sources, geometry=geometry_geo_sources, crs='EPSG:4326')

######################## geo sources - shp file ##############################
# output_filename = "geo_sources.shp"
# output_filepath = os.path.join(output_path, output_filename)
# geo_df_geo_sources.to_file(output_filepath,encoding='ISO-8859-1' )

############################################# WASTE HEAT from industry (HT e LT) and from waste water trestment plants ###############################################

df_wwpp = pd.read_excel(input_filepath, sheet_name='WWTP')
df_ht = pd.read_excel(input_filepath, sheet_name='HT')
df_lt = pd.read_excel(input_filepath, sheet_name='LT')

# Concatenate the three DataFrames
df_2 = pd.concat([df_wwpp, df_ht, df_lt], ignore_index=True)
geometry_2 = gpd.points_from_xy(df_2["long"], df_2["lat"])
geo_df_fonti = gpd.GeoDataFrame(df_2, geometry=geometry_2)
geo_df_fonti.crs = 'EPSG:4326'
# print(geo_df_2.head())

######################## heat sources - shp file ##############################
# output_filename = "heat_sources.shp"
# output_filepath = os.path.join(output_path, output_filename)
# geo_df_fonti.to_file(output_filepath,encoding='ISO-8859-1' )

############################################################## MUNICIPALITIES ######################################################################

# Read the shapefile of municipalities
input_filename = 'Com01012021_WGS84.shp'
input_filepath = os.path.join(input_path, input_filename)
comuni_gdf = gpd.read_file(input_filepath, encoding='utf-8', dtype={'PRO_COM_T': str})
comuni_gdf = comuni_gdf.to_crs(epsg=4326)

# Create a GeoDataFrame with municipalities and centroids
comuni_gdf_con_centroidi = gpd.GeoDataFrame({
    'COMUNE': comuni_gdf['COMUNE'],
    'PRO_COM_T': comuni_gdf['PRO_COM_T'],
    'y_centroide': [geom.centroid.coords[0][0] for geom in comuni_gdf['geometry']],
    'x_centroide': [geom.centroid.coords[0][1] for geom in comuni_gdf['geometry']]
}, geometry=[Point(geom.centroid.coords[0]) for geom in comuni_gdf['geometry']], crs='EPSG:4326')

######################## micipatilities centroids - shp file ##############################
# output_filename = "municipalities_centroids.shp"
# output_filepath = os.path.join(output_path, output_filename)
# comuni_gdf_con_centroidi.to_file(output_filepath, encoding='utf-8')

############################################################# BUFFER ###########################################################

# merge geodf
merged_geo_df = gpd.GeoDataFrame(pd.concat([geo_df_dmd, geo_df_fonti , geo_df_geo_sources], ignore_index=True))

#Extract the coordinates of the points
points = merged_geo_df.geometry.unary_union
x_coords = [point.x for point in points.geoms]
y_coords = [point.y for point in points.geoms]
points_array = list(zip(x_coords, y_coords))

# Function to create a buffer around a source
def create_buffer_around_source(source_geometry, radius):
    return source_geometry.buffer(radius)

############### USER INPUT ######################################################

from shapely.geometry import Point

comuni_disponibili = comuni_gdf_con_centroidi['COMUNE'].unique()
print("Available municipalities:", comuni_disponibili)

comune_scelto = input("Choose a municipality: ")
fonte_di_interesse = comuni_gdf_con_centroidi[comuni_gdf_con_centroidi['COMUNE'] == comune_scelto].geometry.squeeze()

raggio_in_metri = float(input("Enter the radius in meters: "))

lat, lon = fonte_di_interesse.y, fonte_di_interesse.x
raggio_in_gradi = raggio_in_metri / geodesic((lat, lon), (lat + 1, lon)).meters
area_di_interesse = fonte_di_interesse.buffer(raggio_in_gradi)

########################################## ROUTING  - crea percorsi stradali che collegano i punti più vicini su percorso stradale ai punti all'interno 

geo_df_filtrato = merged_geo_df[merged_geo_df.geometry.within(area_di_interesse)]
points_array = np.array([(point.x, point.y) for point in geo_df_filtrato.geometry])
buffer_polygon = area_di_interesse
graph = ox.graph_from_polygon(buffer_polygon, network_type='drive')
origin_node = ox.distance.nearest_nodes(graph, fonte_di_interesse.x, fonte_di_interesse.y)
destination_nodes = [ox.distance.nearest_nodes(graph, point.x, point.y) for point in geo_df_filtrato.geometry]
routes = [ox.shortest_path(graph, origin_node, dest_node, weight='length') for dest_node in destination_nodes]

segments = []

for route in routes:
    if route is not None:
   
        for i in range(len(route) - 1):
          
            start_coords = (graph.nodes[route[i]]['x'], graph.nodes[route[i]]['y'])
            end_coords = (graph.nodes[route[i + 1]]['x'], graph.nodes[route[i + 1]]['y'])

            segment_line = LineString([start_coords, end_coords])

            segments.append({'geometry': segment_line,
                             'lat_start': start_coords[1], 'lon_start': start_coords[0],
                             'lat_arrive': end_coords[1], 'lon_arrive': end_coords[0]})
    else:
        print("Il percorso non è valido: None")


segments_gdf = gpd.GeoDataFrame(segments, geometry='geometry', crs='EPSG:4326')

######################## Road path segments - shp file ##############################

# output_filename = "road_path.shp"
# output_filepath = os.path.join(output_path, output_filename)
# segments_gdf.to_file(output_filepath, encoding= 'utf-8')

########################################## This part of the code extracts the coordinates of nodes along the road path 
########################################## and connects the sources and demands to the nearest road node

valid_coordinates_df = geo_df_filtrato[['latitude', 'longitude', 'lat', 'long']].T.ffill().T
valid_coordinates_df = valid_coordinates_df.dropna(subset=['latitude', 'longitude', 'lat', 'long'], how='all')
new_df = pd.DataFrame({'LAT': valid_coordinates_df['latitude'].combine_first(valid_coordinates_df['lat']),
                       'LONG': valid_coordinates_df['longitude'].combine_first(valid_coordinates_df['long'])})

nodes_df = segments_gdf[['lat_start', 'lon_start', 'lat_arrive', 'lon_arrive']].copy()
nodes_df.columns = ['lat', 'lon', 'lat_arrive', 'lon_arrive']
nodes_df = pd.concat([nodes_df[['lat', 'lon']], nodes_df[['lat_arrive', 'lon_arrive']].rename(columns={'lat_arrive': 'lat', 'lon_arrive': 'lon'})])
nodes_df = nodes_df.drop_duplicates()
nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=gpd.points_from_xy(nodes_df['lon'], nodes_df['lat']))
nodes_gdf.to_file("strada_nodes.shp")

nodes_array = np.array([(x, y) for x, y in zip(nodes_gdf['lat'], nodes_gdf['lon'])])

def find_nearest_node(point, nodes):
    tree = cKDTree(nodes)
    _, idx = tree.query(point)
    return idx

new_segments = []

for idx, point_row in new_df.iterrows():
    nearest_node_idx = find_nearest_node((point_row['LAT'], point_row['LONG']), nodes_array)
    new_segment = {'geometry': LineString([(nodes_gdf.iloc[nearest_node_idx]['lon'], nodes_gdf.iloc[nearest_node_idx]['lat']),
                                           (point_row['LONG'], point_row['LAT'])]),
                   'lat_start': nodes_gdf.iloc[nearest_node_idx]['lat'],
                   'lon_start': nodes_gdf.iloc[nearest_node_idx]['lon'],
                   'lat_arrive': point_row['LAT'],
                   'lon_arrive': point_row['LONG']}
    new_segments.append(new_segment)

# Concatenate the new segments to the existing GeoDataFrame
segments_gdf = gpd.GeoDataFrame(pd.concat([segments_gdf, gpd.GeoDataFrame(new_segments, geometry='geometry')], ignore_index=True))



##############################################################################################################
##############################################################################################################

segments_gdf['id_start'] = segments_gdf['lat_start'].astype(str) + '_' + segments_gdf['lon_start'].astype(str)
segments_gdf['id_arrive'] = segments_gdf['lat_arrive'].astype(str) + '_' + segments_gdf['lon_arrive'].astype(str)

# Filter the GeoDataFrame geo_df_dmd to keep only geometries that are entirely within the area of interest
geo_df_dmd_buffer = geo_df_dmd[geo_df_dmd.geometry.within(area_di_interesse)]
geo_df_fonti_buffer = geo_df_fonti[geo_df_fonti.geometry.within(area_di_interesse)]
geo_df_geo_sources_buffer = geo_df_geo_sources[geo_df_geo_sources.geometry.within(area_di_interesse)]

################################# geodataframe shp ###########################################

output_filename = "dmd_in_buffer.shp"
output_filepath = os.path.join(output_path, output_filename)
geo_df_dmd_buffer.to_file(output_filepath, encoding= 'utf-8')

########

output_filename = "sources_in_buffer.shp"
output_filepath = os.path.join(output_path, output_filename)
geo_df_fonti_buffer.to_file(output_filepath, encoding='utf-8')

########

output_filename = "geo_sources_in_buffer.shp"
output_filepath = os.path.join(output_path, output_filename)
geo_df_geo_sources_buffer.to_file("fonti_geo_buffer.shp", encoding='utf-8')

#######################################################################################

geo_df_dmd_buffer['id_dmd'] = geo_df_dmd_buffer['latitude'].astype(str) + '_' + geo_df_dmd_buffer['longitude'].astype(str)
geo_df_fonti_buffer['id_source'] = geo_df_fonti_buffer['lat'].astype(str) + '_' + geo_df_fonti_buffer['long'].astype(str)
geo_df_geo_sources_buffer['id_geo_source'] = geo_df_geo_sources_buffer['latitude'].astype(str) + '_' + geo_df_geo_sources_buffer['longitude'].astype(str)
target_epsg = 'EPSG:32633' 
segments_gdf = segments_gdf.to_crs(target_epsg)
segments_gdf['length_meters'] = segments_gdf['geometry'].length

######################## Road path segments_complete - shp file ##############################
# output_filename = "segments_shapefile_updated.shp"
# output_filepath = os.path.join(output_path, output_filename)
# segments_gdf.to_file(output_filepath, encoding= 'utf-8')


########################  OPTIMIZATION MODEL ##########################


import oemof.solph as solph
from oemof.tools import logger
tec_data = pd.read_excel("data_02.xlsx", sheet_name = 'TEC')
commodities_data = pd.read_excel("data_02.xlsx", sheet_name = 'COMMODITIES')

id_dmd_set = set(geo_df_dmd_buffer['id_dmd'].tolist())
id_fonti_set = set(geo_df_fonti_buffer['id_source'])

id_nodes_set = set(segments_gdf['id_start'].tolist() )
update = set(segments_gdf['id_arrive']) - id_nodes_set
id_nodes_set.update(update)

o_nodes = {}

##################################################

for node in id_nodes_set:
    o_nodes[node] = solph.Bus(label =  node)

for i, c in commodities_data.iterrows():
  

    o_nodes [c['commodity']] = solph.Bus (label = c['commodity'])
   
    if c ['supply']>0:
        label = 'supply_' + c['commodity']
        o_nodes[label] = solph.components.Source(label = label, 
                        outputs={
                        o_nodes[c['commodity']]: solph.Flow(
                                            variable_costs = c['supply_cost_eur_MWh'])})
    
    else:
        
        label = 'sales_' + c['commodity']
        o_nodes[label] = solph.components.Sink(label = label, 
                        inputs={
                        o_nodes[c['commodity']]: solph.Flow(
                                            variable_costs = c['sales_cost_eur_MWh'])})

################################ heat transport network  #############################

for i, b in segments_gdf.iterrows():
    

        label= b['id_start'] + '__' +  b['id_arrive']
        if label not in o_nodes:
            o_nodes[label] = solph.components.Converter(
            label=label,
            inputs={
            o_nodes[b['id_start']]: solph.Flow()
                  },
            outputs={
            o_nodes[b['id_arrive']]: solph.Flow(
                    variable_costs = b['length_meters']*(tec_data.loc[tec_data['technology'] == 'transport', 'cost_euro_MWh'].values[0]))
                  },
            conversion_factors={
                o_nodes[b['id_arrive']]: (tec_data.loc[tec_data['technology'] == 'transport', 'eff'].values[0])},
                )

        rev_label= b['id_arrive'] + '__' +  b['id_start']
        if  rev_label  not in o_nodes:
            o_nodes[rev_label] = solph.components.Converter(
            label=rev_label,
            inputs={
            o_nodes[b['id_arrive']]: solph.Flow()
                          },
            outputs={
            o_nodes[b['id_start']]: solph.Flow(
                    variable_costs = b['length_meters']*(tec_data.loc[tec_data['technology'] == 'transport', 'cost_euro_MWh'].values[0]))
                          },
                   conversion_factors={
                       o_nodes[b['id_start']]:(tec_data.loc[tec_data['technology'] == 'transport', 'eff'].values[0])})
                       
########################## energy demands ##############################################

for i, s in geo_df_dmd_buffer.iterrows():
                  label= s ['id_dmd'] + '_' + 'dmd'
                  o_nodes[label] = solph.components.Sink(label = label, 
                  inputs={
                  o_nodes[s['id_dmd']] : solph.Flow(
                  nominal_value = s['PTE_tot [MWh/anno]'],
                  fix = 1)})
                  
##################### HT, LT and WWTP sources ##########################################  
               
for i, s in geo_df_fonti_buffer.iterrows():

   if s['Tipo'] == 'HT':
         label= s['id_source'] + '_' + 'HT'
         o_nodes[label] = solph.components.Source(label = label, 
         outputs={
              o_nodes[s['id_source']] : solph.Flow(
              nominal_value = s['Energia [MWh]'],
              variable_costs = s['Costo [EUR/MWh]'])})
                         
   if s['Tipo'] == 'LT':
        label= s['id_source'] + '_' + 'LT'
        o_nodes[label] = solph.components.Source(label = label, 
        outputs={
             o_nodes[s['id_source']] : solph.Flow(
             nominal_value = s['Energia [MWh]'],
             variable_costs = s['Costo [EUR/MWh]'])}) 

   if s['Tipo'] == 'WWTP':
          label= s['id_source'] + '_' + 'WWTP'
          o_nodes[label] = solph.components.Source(label = label, 
          outputs={
               o_nodes[s['id_source']] : solph.Flow(
               nominal_value = s['Energia [MWh]'],
               variable_costs = s['Costo [EUR/MWh]'])}) 
                   
####################### geo sources ##################################################

for i, s in geo_df_geo_sources_buffer.iterrows():
    
    if s['likely_geo'] > 0:
        label= s['id_geo_source'] + '_' +  'geo_shallow'
        o_nodes[label] = solph.components.Source(label = label, 
        outputs={
             o_nodes[s['id_geo_source']] : solph.Flow(
                   
             nominal_value = s['l_geo_MWh'],
             variable_costs = s['costo_E_MWh']
                 
                         )})         

    elif s['geo_1000_2000'] > 0:
        label= s['id_geo_source'] + '_' +  'geo_1000_2000'
        o_nodes[label] = solph.components.Source(label = label, 
        outputs={
             o_nodes[s['id_geo_source']] : solph.Flow(
                   
             nominal_value = s['geo_1000_2000_MWh'],
             variable_costs = s['costo_E_MWh'])})
                 
                                 
##################################### CHP #####################################

for node in id_fonti_set:
    label = node + '_' + 'chp'
    o_nodes[label] = solph.components.Converter(
    label=label,
    inputs={
    o_nodes['bus_gas']: solph.Flow()
                  },
    outputs={
    o_nodes[node]: solph.Flow(
            variable_costs = (tec_data.loc[tec_data['technology'] == 'CHP', 'cost_euro_MWh'].values[0]),
            nominal_value = 0)
                  ,
    o_nodes['bus_elc_out']: solph.Flow(
            variable_costs = commodities_data.loc[commodities_data['commodity'] == 'bus_elc_out', 'sales_cost_eur_MWh'].values[0],
            nominal_value = 25000  )},
            conversion_factors={
                o_nodes[node]:(tec_data.loc[tec_data['technology'] == 'CHP', 'eff'].values[0]), 
                o_nodes['bus_elc_out']:(tec_data.loc[tec_data['technology'] == 'CHP', 'eff_e'].values[0])
                })

##################### demand technologies #########################################

for i, t in tec_data.iterrows():
    for dmd in id_dmd_set:
        if t['type'] == 'dmd':
            
            label= dmd + '_' + t['technology']
            o_nodes[label] = solph.components.Converter(
            label=label,
            inputs={
            o_nodes[t['bus_in']]: solph.Flow(variable_costs = t['delta_comb_eur_MWh'])
                          },
            outputs={
            o_nodes[dmd]: solph.Flow(
                    variable_costs = t['cost_euro_MWh'])
                          },
                   conversion_factors={
                       o_nodes[dmd] : t['eff'] })
       
print("********************************************************")
print("The following objects has been created from excel sheet:")
for n in o_nodes:
    print(n)       
        
        
logger.define_logging()
datetime_index = pd.date_range(
    '1/1/2013', periods = 1, freq='H')
es = solph.EnergySystem(timeindex=datetime_index)
es.add(*o_nodes.values())
logger.define_logging()
om = solph.Model(es)
om.solve(solver='glpk')

logging.info("processing results")


results = solph.processing.results(om)
result = solph.processing.convert_keys_to_strings(results)
parameters = solph.processing.parameter_as_dict(om, exclude_none=True) 
parameter = solph.processing.convert_keys_to_strings(parameters)

logging.info("writing results")       

################################################################################################################################
###################################################### RESULTS #################################################################
################################################################################################################################

satisfied_demands ={}
for r in result:
    res = result[r]['sequences'].sum()
    par = parameter[r]['scalars']
    satisfied_demands.update({r: (r, res.get('flow'))})
satisfied_demands = {k: v for k, v in satisfied_demands.items() if v[0] != None}  #if v[1]>0}
satisfied_demands_dataframe = pd.DataFrame(satisfied_demands)
satisfied_demands_dataframe = satisfied_demands_dataframe.transpose()

output_filename = "raw_results"
output_filepath = os.path.join(output_path, output_filename)
satisfied_demands_dataframe[1].to_csv(output_filepath) 

risultati = satisfied_demands_dataframe.copy()
risultati = satisfied_demands_dataframe.reset_index()
risultati.columns = ['node_1', 'node_2',  'tupla' , 'flow']


######################### DH PATH ########################

condizione = (risultati['node_1'].str.contains("__")) | (risultati['node_2'].str.contains("__")) & (risultati['flow'] > 0)

if condizione.any():
    risultati_tracciato = risultati[condizione  & (risultati['flow'] > 0)]

    if not risultati_tracciato.empty:
        coordinate_tracciato = risultati_tracciato['node_1'].str.split('__|_', expand=True).combine_first(risultati_tracciato['node_2'].str.split('__|_', expand=True))
        coordinate_tracciato.columns = ['lat_start', 'long_start', 'lat_arrive', 'long_arrive']
        coordinate_tracciato = coordinate_tracciato.drop_duplicates()
        geometrie_linee = [LineString([(row['long_start'], row['lat_start']), (row['long_arrive'], row['lat_arrive'])]) 
                               for _, row in coordinate_tracciato.iterrows()]
        
        tracciato_output_gdf = gpd.GeoDataFrame(coordinate_tracciato, 
                                   geometry=geometrie_linee, 
                                   crs='EPSG:4326')  # Assicurati di impostare il sistema di riferimento corretto
        
        output_filename = "district_heating_path.shp"
        output_filepath = os.path.join(output_path, output_filename)
        tracciato_output_gdf.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no district heating')
else:
    print('no district heating')

############################## CHP coordinates  ##########################

condizione_chp = risultati['node_1'].str.endswith("_chp") & (risultati['node_2'] != "bus_elc_out") & (risultati['flow'] > 0)

if condizione_chp.any():
    risultati_chp = risultati[condizione_chp]

    if not risultati_chp.empty:
        coordinate_chp = risultati_chp['node_2'].str.split('_', expand=True)
        coordinate_chp.columns = ['lat_chp', 'long_chp']
        
        geometrie_punti = [Point(float(row['long_chp']), float(row['lat_chp'])) for _, row in coordinate_chp.iterrows()]
        
        gdf_coordinate_chp = gpd.GeoDataFrame(coordinate_chp, 
                                              geometry=geometrie_punti, 
                                              crs='EPSG:4326')
        
        output_filename = "CHP.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_chp.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no chp')
else:
    print('no chp')

############################## boiler ##########################

condizione_boiler = risultati['node_1'].str.endswith("_boiler_gas") & (risultati['flow'] > 0)

if condizione_boiler.any():
    risultati_boiler = risultati[condizione_boiler]
    
    if not risultati_boiler.empty:

        coordinate_boiler = risultati_boiler['node_2'].str.split('_', expand=True)
        coordinate_boiler.columns = ['lat_boiler', 'long_boiler']
        coordinate_boiler['flow'] = risultati_boiler['flow']
        
        geometrie_punti = [Point(float(row['long_boiler']), float(row['lat_boiler'])) for _, row in coordinate_boiler.iterrows()]
        
        gdf_coordinate_boiler = gpd.GeoDataFrame(coordinate_boiler, 
                                                  geometry=geometrie_punti, 
                                                  crs='EPSG:4326')
        
        output_filename = "boiler.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_boiler.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no boiler')
else:
    print('no boiler')
    
############################## heat pumps  ##########################

condizione_pdc_ee = risultati['node_1'].str.endswith("_pdc_ee") & (risultati['flow'] > 0)

if condizione_pdc_ee.any():
    risultati_pdc_ee = risultati[condizione_pdc_ee]
    
    if not risultati_pdc_ee.empty:

        coordinate_pdc_ee = risultati_pdc_ee['node_2'].str.split('_', expand=True)
        coordinate_pdc_ee.columns = ['lat_pdc_ee', 'long_pdc_ee']
        coordinate_pdc_ee['flow'] = risultati_pdc_ee['flow']
        
        geometrie_punti = [Point(float(row['long_pdc_ee']), float(row['lat_pdc_ee'])) for _, row in coordinate_pdc_ee.iterrows()]
        
        gdf_coordinate_pdc_ee = gpd.GeoDataFrame(coordinate_pdc_ee, 
                                                  geometry=geometrie_punti, 
                                                  crs='EPSG:4326')
        
        output_filename = "heat_pumps.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_pdc_ee.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('Il sistema non contiene pdc_ee')
else:
    print('Il sistema non contiene pdc_ee')
    
    
############################## industry waste heat HT ##########################

condizione_HT = risultati['node_1'].str.endswith("_HT") & (risultati['flow'] > 0)

if condizione_HT.any():
    risultati_HT = risultati[condizione_HT]
    
    if not risultati_HT.empty:

        coordinate_HT = risultati_HT['node_2'].str.split('_', expand=True)
        coordinate_HT.columns = ['lat_HT', 'long_HT']
        coordinate_HT['flow'] = risultati_HT['flow']
        
        geometrie_punti = [Point(float(row['long_HT']), float(row['lat_HT'])) for _, row in coordinate_HT.iterrows()]
        
        gdf_coordinate_HT = gpd.GeoDataFrame(coordinate_HT, 
                                              geometry=geometrie_punti, 
                                              crs='EPSG:4326')
        output_filename = "HT.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_HT.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no HT')
else:
    print('no HT')

############################## industry waste heat LT ##########################

condizione_LT = risultati['node_1'].str.endswith("_LT") & (risultati['flow'] > 0)

if condizione_LT.any():
    risultati_LT = risultati[condizione_LT]
    
    if not risultati_LT.empty:
       
        coordinate_LT = risultati_LT['node_2'].str.split('_', expand=True)
        coordinate_LT.columns = ['lat_LT', 'long_LT']
        coordinate_LT['flow'] = risultati_LT['flow']
        
        geometrie_punti = [Point(float(row['long_LT']), float(row['lat_LT'])) for _, row in coordinate_LT.iterrows()]
        
        gdf_coordinate_LT = gpd.GeoDataFrame(coordinate_LT, 
                                              geometry=geometrie_punti, 
                                              crs='EPSG:4326')
        
        output_filename = "LT.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_LT.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no LT')
else:
    print('no LT')

############################## WWTP ##########################

condizione_WWTP = risultati['node_1'].str.endswith("_WWTP") & (risultati['flow'] > 0)

if condizione_WWTP.any():
    
    risultati_WWTP = risultati[condizione_WWTP]
    
    if not risultati_WWTP.empty:
        
        coordinate_WWTP = risultati_WWTP['node_2'].str.split('_', expand=True)
        coordinate_WWTP.columns = ['lat_WWTP', 'long_WWTP']
        coordinate_WWTP['flow'] = risultati_WWTP['flow']
        
        geometrie_punti = [Point(float(row['long_WWTP']), float(row['lat_WWTP'])) for _, row in coordinate_WWTP.iterrows()]
        
        gdf_coordinate_WWTP = gpd.GeoDataFrame(coordinate_WWTP, 
                                               geometry=geometrie_punti, 
                                               crs='EPSG:4326')
        
        output_filename = "WWTP.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_WWTP.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no WWTP')
else:
    print('no WWTP')

############################## geo_shallow ##########################

condizione_geo_shallow = risultati['node_1'].str.endswith("_geo_shallow") & (risultati['flow'] > 0)

if condizione_geo_shallow.any():

    risultati_geo_shallow = risultati[condizione_geo_shallow]
    
    if not risultati_geo_shallow.empty:
       
        coordinate_geo_shallow = risultati_geo_shallow['node_2'].str.split('_', expand=True)
        coordinate_geo_shallow.columns = ['lat_geo_shallow', 'long_geo_shallow']
        coordinate_geo_shallow['flow'] = risultati_geo_shallow['flow']
        
        geometrie_punti = [Point(float(row['long_geo_shallow']), float(row['lat_geo_shallow'])) for _, row in coordinate_geo_shallow.iterrows()]
        
        gdf_coordinate_geo_shallow = gpd.GeoDataFrame(coordinate_geo_shallow, 
                                                        geometry=geometrie_punti, 
                                                        crs='EPSG:4326')
        
        output_filename = "geo_shallow.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_geo_shallow.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no geo_shallow')
else:
    print('no geo_shallow')
    
    
##############################  geo_1000_2000 ##########################

condizione_geo_1000_2000 = risultati['node_1'].str.endswith("_geo_1000_2000") & (risultati['flow'] > 0)

if condizione_geo_1000_2000.any():

    risultati_geo_1000_2000 = risultati[condizione_geo_1000_2000]
    
    if not risultati_geo_1000_2000.empty:

        coordinate_geo_1000_2000 = risultati_geo_1000_2000['node_2'].str.split('_', expand=True)
        coordinate_geo_1000_2000.columns = ['lat_geo_1000_2000', 'long_geo_1000_2000']
        coordinate_geo_1000_2000['flow'] = risultati_geo_1000_2000['flow']
        
        geometrie_punti = [Point(float(row['long_geo_1000_2000']), float(row['lat_geo_1000_2000'])) for _, row in coordinate_geo_1000_2000.iterrows()]
        
        gdf_coordinate_geo_1000_2000 = gpd.GeoDataFrame(coordinate_geo_1000_2000, 
                                                         geometry=geometrie_punti, 
                                                         crs='EPSG:4326')
        
        output_filename = "geo_1000_2000.shp"
        output_filepath = os.path.join(output_path, output_filename)
        gdf_coordinate_geo_1000_2000.to_file(output_filepath, driver='ESRI Shapefile')
    else:
        print('no geo_1000_2000')
else:
    print('no  geo_1000_2000')

##################################################### INTERACTIVE MAP ######################################################

tracciato_gdf = gpd.read_file('tracciato_output.shp')
coordinate_gdf_list = [
    ('coordinate_chp.shp', 'CHP', 'blue', 'bolt'),
    ('coordinate_boiler.shp', 'Boiler', 'red', 'fire'),
    ('coordinate_pdc_ee.shp', 'PDC_EE', 'green', 'leaf'),
    ('coordinate_HT.shp', 'HT', 'purple', 'cloud'),
    ('coordinate_LT.shp', 'LT', 'orange', 'tint'),
    ('coordinate_WWTP.shp', 'WWTP', 'gray', 'tint'),
    ('coordinate_geo_shallow.shp', 'Geo Shallow', 'yellow', 'tint'),
    ('coordinate_geo_1000_2000.shp', 'Geo 1000-2000', 'brown', 'tint'),
]


center = [tracciato_gdf.total_bounds[1] + (tracciato_gdf.total_bounds[3] - tracciato_gdf.total_bounds[1]) / 2,
          tracciato_gdf.total_bounds[0] + (tracciato_gdf.total_bounds[2] - tracciato_gdf.total_bounds[0]) / 2]


mappa = folium.Map(location=center, zoom_start=12)

folium.GeoJson(tracciato_gdf, weight=5).add_to(mappa)

marker_cluster = MarkerCluster().add_to(mappa)

for coordinate_file, label, color, icon in coordinate_gdf_list:
    if not os.path.exists(coordinate_file):
        print(f"File not found: {coordinate_file}. Skipping...")
        continue

    try:
        coordinate_gdf = gpd.read_file(coordinate_file)
    except Exception as e:
        print(f"Error reading {coordinate_file}: {str(e)}. Skipping...")
        continue

    for idx, row in coordinate_gdf.iterrows():
       
        if 'flow' in row.index and (pd.isnull(row['flow']) or row['flow'] == 0):
            continue
        try:
            flow_value = f"{float(row['flow']):.2f}" if 'flow' in row.index else "N/A"
        except ValueError:
            flow_value = "N/A"
        popup_text = f"{label}<br>Flow: {flow_value} MWh"
        folium.Marker(location=[row.geometry.y, row.geometry.x], popup=popup_text,
                      icon=folium.Icon(color=color, icon=icon)).add_to(marker_cluster)

output_filename = 'interactive_map.html'
output_filepath = os.path.join(output_path, output_filename)
mappa.save(output_filepath)


output_filename = 'results_table.xlsx'
output_filepath = os.path.join(output_path, output_filename)

with pd.ExcelWriter(output_filepath) as w: 
    if 'tracciato_output_gdf' in globals():
        tracciato_output_gdf.to_excel(w, sheet_name='DH_path')
    if 'gdf_coordinate_boiler' in globals():
        gdf_coordinate_boiler.to_excel(w, sheet_name='boiler')
    if 'gdf_coordinate_pdc_ee' in globals():
        gdf_coordinate_pdc_ee.to_excel(w, sheet_name='heat_pumps')
    if 'gdf_coordinate_HT' in globals():
        gdf_coordinate_HT.to_excel(w, sheet_name='HT')
    if 'gdf_coordinate_LT' in globals():
        gdf_coordinate_LT.to_excel(w, sheet_name='LT')
    if 'gdf_coordinate_WWTP' in globals():
        gdf_coordinate_WWTP.to_excel(w, sheet_name='WWTP')
    if 'gdf_coordinate_geo_shallow' in globals():
        gdf_coordinate_geo_shallow.to_excel(w, sheet_name='geo_shallow')
    if 'gdf_coordinate_geo_1000_2000' in globals():
        gdf_coordinate_geo_1000_2000.to_excel(w, sheet_name='geo_1000_2000')