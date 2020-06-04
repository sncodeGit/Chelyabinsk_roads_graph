# -*- coding: utf-8 -*-

# Необходимые модули
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import osmnx as ox
import numpy as np
import xml.etree.ElementTree as ET

# Дополнительные скрипты
import config

def convert_to_graph(osm_file = config.osm_file):
    graph = ox.graph_from_file(osm_file, simplify= False, retain_all= True, name= 'Chel')
    largest_component = list(max(nx.strongly_connected_components(graph), key=len))
    return graph, largest_component


def get_ids(type_object, osm_file = config.osm_file, category = 'amenity'):
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
            if i['k'] == category and i['v'] == type_object:
                result.append(np.random.choice(refs))
    return list(set(result))

def get_nodes_and_objs():
    objs= get_ids('school')
    objs+= get_ids('hospital', category='healthcare')

    nodes = get_ids('apartments', category= 'building')
    nodes += get_ids(get_ids('detached', category='building'))
    return set(objs), set(nodes)

def get_random_of_objecs(lst, N, largest_component = None):
    lst = np.array(lst)
    if largest_component:
        new_lst = lst[np.isin(lst, largest_component)]
    else:
        new_lst = lst
    res = np.random.choice(new_lst, N, replace = False)
    return list(res)

def get_highways(osm_file = config.osm_file):
    root = ET.parse(osm_file).getroot()
    nodes = root.findall('./way')
    result = []

    for node in nodes:
        refs = []
        f = False
        for param in node:
            if 'ref' in param.attrib.keys():
                refs.append(int(param.attrib['ref']))
            else:
                if param.attrib['k'] == 'highway':
                    f = True
            if f:
                result.append(refs)
    result = [i for sub in result for i in sub]
    return (result)


def corrected_objects(graph, objects_list, points_list):
    points_list = np.array(points_list)

    simp_dict = dict(graph.nodes(data=True))
    obj_coords = np.array([list(simp_dict[key].values())[:2] for key in objects_list])
    point_coords = np.array([list(simp_dict[key].values())[:2] for key in points_list])

    nearest_ids = [np.argmin(np.linalg.norm(point_coords - oc, axis=1)) for oc in obj_coords]
    corrected_objects = points_list[nearest_ids]

    return list(set(corrected_objects))


def get_resulting_graph(graph, objects_list, nodes_list, visualize, strong_components=False, isolates=False):
    graph_simp = ox.simplify.simplify_graph(graph)
    print(len(nodes_list))

    graph_dict = dict(graph.nodes(data=True))
    df_obj = pd.DataFrame({node: graph_dict[node] for node in objects_list if node in graph_dict}).T
    print('Nodes number be4 adding:', len(graph_simp.nodes))
    print('df_obj:', len(df_obj))
    df_nodes = pd.DataFrame({node: graph_dict[node] for node in nodes_list if node in graph_dict}).T
    print('df_nodes:', len(df_nodes))
    for simp_path in ox.simplify.get_paths_to_simplify(graph):
        for osmid in simp_path:
            argwhere = np.argwhere(df_obj['osmid'].to_numpy() == osmid)
            if argwhere.size > 0 and osmid not in list(graph_simp.nodes):
                i = argwhere[0][0]
                obj_node = df_obj.iloc[i].to_dict()
                graph_simp.add_node(osmid, **obj_node)
                graph_simp.add_edge(simp_path[0], osmid)
                graph_simp.add_edge(osmid, simp_path[-1])

    print('Nodes number after objects adding:', len(graph_simp.nodes))
    print('Objects inters:', len(set(objects_list).intersection(set(graph_simp.nodes()))))

    for simp_path in ox.simplify.get_paths_to_simplify(graph):
        for osmid in simp_path:
            argwhere = np.argwhere(df_nodes['osmid'].to_numpy() == osmid)
            if argwhere.size > 0 and osmid not in list(graph_simp.nodes):
                i = argwhere[0][0]
                obj_node = df_nodes.iloc[i].to_dict()
                graph_simp.add_node(osmid, **obj_node)
                graph_simp.add_edge(simp_path[0], osmid)
                graph_simp.add_edge(osmid, simp_path[-1])
    print('Nodes number after nodes adding:', len(graph_simp.nodes))
    print('Nodes inters:', len(set(nodes_list).intersection(set(graph_simp.nodes()))))

    if strong_components:
        for component in list(nx.strongly_connected_components(graph_simp)):
            if len(component) < 100:
                for node in component:
                    to_remove = []
                    if node not in (nodes_list + objects_list):
                        to_remove.append(node)
                    else:
                        break
                    graph_simp.remove_nodes_from(to_remove)

    print(len(graph_simp.nodes))

    if isolates:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    if visualize:
        oc = ['r' if osmid in objects_list else 'g' if osmid in nodes_list
        else 'b' for osmid in graph_simp.nodes()]
        os = [200 if osmid in objects_list or osmid in nodes_list
              else 10 for osmid in graph_simp.nodes()]
        fig, ax = ox.plot_graph(graph_simp, node_color=oc, node_size=os, fig_height=18)
        pos = {}
        for key in list(graph_dict.keys()):
            pos[key] = (graph_dict[key]['x'], graph_dict[key]['y'])
        labels = {}
        for i in range(len(objects_list)):
            labels[objects_list[i]] = 'O' + str(i)
        for j in range(len(nodes_list)):
            labels[nodes_list[j]] = 'N' + str(j)
        plt.figure(figsize=(30, 18))
        nc = ['r' if nid in objects_list else 'g' for nid in objects_list + nodes_list]
        nx.draw_networkx(graph, pos=pos, nodelist=objects_list + nodes_list, node_color=nc, with_labels=False, arrows=False,
                         node_size=350, edge_color='gray')
        nx.draw_networkx_labels(graph, pos=pos, labels=labels, font_size=10)
        plt.savefig('Chosen_nodes_and_objects.png')

    return graph_simp

#
# G, LC = convert_to_graph()
# obj, nodes = get_nodes_and_objs()
# road_points = get_highways()
# c_obj = corrected_objects(G, obj, road_points)
# c_nodes = corrected_objects(G, nodes, road_points)
#
# c_obj = get_random_of_objecs(c_obj, 10, largest_component=LC)
# c_nodes = get_random_of_objecs(c_nodes, 100, largest_component=LC)
#
# G = get_resulting_graph(G, c_obj, c_nodes, True, strong_components= True)

# for i in G.edges():
#     print(i)
