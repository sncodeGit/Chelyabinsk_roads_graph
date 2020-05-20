# -*- coding: utf-8 -*-

# Необходимые модули
import networkx as nx
import osmnx as ox
import xml.etree.ElementTree as ET

# Дополнительные скрипты
import config

def convert_to_graph(osm_file = config.osm_file):
    graph = ox.graph_from_file(osm_file, simplify= False, retain_all= True, name= 'Chel')
    return graph
