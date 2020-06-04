# -*- coding: utf-8 -*-

# Необходимые модули
import networkx as nx
import osmnx as ox
import numpy as np
import xml.etree.ElementTree as ET

# Дополнительные скрипты
import config

# def convert_to_graph(osm_file = config.osm_file):
#     graph = ox.graph_from_file(osm_file, simplify= False, retain_all= True, name= 'Chel')
#     return graph

# G = convert_to_graph()
# for i in G.edges():
#     print(i)

def get_ids(type_object, osm_file = config.osm_file):
    root = ET.parse(osm_file).getroot()
    nodes = root.findall('./way')
    result = []

    for node in nodes:
        refs = []
        tags = []
        for param in node:
            if 'ref' in param.attrib.keys():
                refs.append(int(param.attrib['ref']))
            else:
                tags.append({'k':param.attrib['k'], 'v':param.attrib['v']})
        for i in tags:
            if i['k'] == 'amenity' and i['v'] == type_object:
                result.append(np.random.choice(refs))
    return list(set(result))
