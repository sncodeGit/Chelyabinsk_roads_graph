import pandas as pd
import converting_to_csv
import config
import numpy as np
from heapq import heappush, heappop
import osmnx as ox
import shutil
import os
import sys

def dijkstra(graph: pd.DataFrame, edges: dict, src: int, dst: set):
    mins = {vertex: {'cost': np.inf, 'path': ()} for vertex in graph.index}
    mins[src] = {'cost': 0, 'path': (src, src)}
    seen = set()
    dst_control = dst.copy()

    queue = [(0, src, ())]
    while queue and dst_control:
        (cost1, vertex1, path) = heappop(queue)

        if vertex1 not in seen:
            seen.add(vertex1)
            dst_control.discard(vertex1)
            path = path + tuple([vertex1])

            for cost2, vertex2 in edges.get(vertex1, ()):
                if vertex2 in seen:
                    continue

                prev = mins.get(vertex2, None)['cost']
                curr = cost1 + cost2

                if prev is None or curr < prev:
                    mins[vertex2] = {
                        'cost': curr,
                        'path': path + tuple([vertex2])
                    }
                    heappush(queue, (curr, vertex2, path))

    mins = {vertex: cost_path for vertex, cost_path in mins.items() if vertex in dst}
    return mins


def get_path(graph, graph_df, graph_edges, node_num, objects_list, nodes_list, output_path=config.path):
    res_df = pd.DataFrame(columns=['type', 'src', 'dst', 'cost', 'path'])

    res_dict_from = {}
    from_dict = dijkstra(graph_df, graph_edges, nodes_list[node_num], set(objects_list))
    min_cost = sys.maxsize
    oid_from = 0
    way_from = ()
    for k in list(from_dict.keys()):
        if from_dict[k]['cost'] < min_cost:
            min_cost = from_dict[k]['cost']
            oid_from = k
            way_from = from_dict[k]['path']
    res_dict_from['from_node'] = {objects_list.index(oid_from): min_cost}

    oid_from_num = objects_list.index(oid_from)

    row_from = {'type': 'from', 'src': node_num, 'dst': oid_from_num,
                'cost': res_dict_from['from_node'][objects_list.index(oid_from)], 'path': way_from}
    res_df = res_df.append(row_from, ignore_index=True)

    to_dict = {}
    for oid in objects_list:
        to_dict[oid] = dijkstra(graph_df, graph_edges, oid, set(nodes_list))[nodes_list[node_num]]
    min_cost = sys.maxsize
    oid_to = 0
    way_to = ()
    for k in list(to_dict.keys()):
        if to_dict[k]['cost'] < min_cost:
            min_cost = to_dict[k]['cost']
            oid_to = k
            way_to = to_dict[k]['path']

    oid_to_num = objects_list.index(oid_to)
    res_dict_from['to_node'] = {objects_list.index(oid_to): min_cost}

    row_to = {'type': 'to', 'src': node_num, 'dst': oid_to_num,
              'cost': res_dict_from['to_node'][objects_list.index(oid_to)], 'path': way_to}
    res_df = res_df.append(row_to, ignore_index=True)

    sum_dict = {}
    for k in list(from_dict.keys()):
        sum_dict[k] = to_dict[k]['cost'] + from_dict[k]['cost']
    min_cost = sys.maxsize
    oid_sum = 0
    for k in list(sum_dict.keys()):
        if sum_dict[k] < min_cost:
            min_cost = sum_dict[k]
            oid_sum = k
    res_dict_from['sum'] = {objects_list.index(oid_sum): min_cost}

    oid_sum_num = objects_list.index(oid_sum)

    row_sum = {'type': 'sum_from', 'src': node_num, 'dst': oid_sum_num,
               'cost': res_dict_from['sum'][objects_list.index(oid_sum)], 'path': from_dict[oid_sum]['path']}
    res_df = res_df.append(row_sum, ignore_index=True)

    row_sum = {'type': 'sum_to', 'src': node_num, 'dst': oid_sum_num,
               'cost': res_dict_from['sum'][objects_list.index(oid_sum)], 'path': to_dict[oid_sum]['path']}

    res_df = res_df.append(row_sum, ignore_index=True)

    from_node_subg = graph.subgraph(from_dict[oid_from]['path'])
    from_edges = list(from_node_subg.edges)

    to_node_subg = graph.subgraph(to_dict[oid_to]['path'])
    to_edges = list(to_node_subg.edges)

    sum_subg_from = graph.subgraph(from_dict[oid_sum]['path'])
    sum_edges_from = list(sum_subg_from.edges)

    sum_subg_to = graph.subgraph(to_dict[oid_sum]['path'])
    sum_edges_to = list(sum_subg_to.edges)

    nc = ['r' if osmid in [oid_from, oid_to, oid_sum] else 'g' if osmid == nodes_list[node_num] else 'gray' for osmid in
          graph.nodes()]
    ns = [400 if osmid in [oid_from, oid_to, oid_sum, nodes_list[node_num]] else 10 for osmid in graph.nodes()]
    ec = [
        'orange' if edge in from_edges else 'yellow' if edge in to_edges else 'purple' if edge in sum_edges_to + sum_edges_from else 'gray'
        for
        edge in list(graph.edges)]
    ew = [5 if edge in from_edges + to_edges else 1 for edge in list(graph.edges)]

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    res_df.to_csv('Output/res1.csv', index=False)
    fig, ax = ox.plot_graph(graph, node_color=nc, node_size=ns, edge_color=ec, edge_linewidth=ew,
                            fig_height=18,
                            save=True, filename='res1')

    return node_num, res_dict_from['from_node'], res_dict_from['to_node'], res_dict_from['sum']


def get_filtered_paths(graph, graph_df, graph_edges, node_num, objects_list, nodes_list, x_limit,
                       output_path=config.path):
    res_df = pd.DataFrame(columns=['type', 'range', 'src', 'dst', 'cost', 'path'])

    from_dict = dijkstra(graph_df, graph_edges, nodes_list[node_num], set(objects_list))
    oids_from = []
    ways_from = []
    oids_costs_from = {}

    for k in list(from_dict.keys()):
        if from_dict[k]['cost'] < x_limit:
            oids_from.append(k)
            curr_cost = from_dict[k]['cost']
            curr_num = objects_list.index(k)
            oids_costs_from[curr_num] = curr_cost
            ways_from.append(from_dict[k]['path'])

    edges_from = []
    for way in ways_from:
        temp_subg = graph.subgraph(way)
        temp_edges = list(temp_subg.edges)
        edges_from.append(temp_edges)

        # print(from_dict)

    for k, v in oids_costs_from.items():
        row_from = {'type': 'from', 'range': x_limit, 'src': node_num, 'dst': k, 'cost': v,
                    'path': from_dict[objects_list[k]]['path']}
        res_df = res_df.append(row_from, ignore_index=True)

        to_dict = {}
        for oid in objects_list:
            to_dict[oid] = dijkstra(graph_df, graph_edges, oid, set(nodes_list))[nodes_list[node_num]]

    oids_to = []
    ways_to = []
    oids_costs_to = {}
    for k in list(to_dict.keys()):
        if to_dict[k]['cost'] < x_limit:
            oids_to.append(k)
            curr_cost = to_dict[k]['cost']
            curr_num = objects_list.index(k)
            oids_costs_to[curr_num] = curr_cost
            ways_to.append(from_dict[k]['path'])

    edges_to = []
    for way in ways_to:
        temp_subg = graph.subgraph(way)
        temp_edges = list(temp_subg.edges)
        edges_to.append(temp_edges)

    for k, v in oids_costs_to.items():
        row_to = {'type': 'to', 'range': x_limit, 'src': node_num, 'dst': k, 'cost': v,
                  'path': to_dict[objects_list[k]]['path']}
        res_df = res_df.append(row_to, ignore_index=True)

    oids_sum = []
    oids_costs_sum = {}

    sum_dict = {}
    for k in list(from_dict.keys()):
        sum_dict[k] = to_dict[k]['cost'] + from_dict[k]['cost']

    for k in list(sum_dict.keys()):
        if sum_dict[k] < x_limit:
            oids_sum.append(k)
            curr_cost = sum_dict[k]
            curr_num = objects_list.index(k)
            oids_costs_sum[curr_num] = curr_cost

    for k, v in oids_costs_sum.items():
        row_sum = {'type': 'sum', 'range': x_limit, 'src': node_num, 'dst': k, 'cost': v,
                   'path': {'from': from_dict[objects_list[k]]['path'],
                            'to': to_dict[objects_list[k]]['path']}}
        res_df = res_df.append(row_sum, ignore_index=True)

    edges_from = [item for sublist in edges_from for item in sublist]
    edges_to = [item for sublist in edges_to for item in sublist]

    nc = ['r' if osmid in oids_from + oids_to else 'g' if osmid == nodes_list[node_num] else 'gray' for osmid in
          graph.nodes()]
    ns = [400 if osmid in oids_from + oids_to or osmid == nodes_list[node_num]
          else 10 for osmid in graph.nodes()]

    ec = ['orange' if edge in edges_from else 'yellow' if edge in edges_to else 'gray' for edge in list(graph.edges)]
    ew = [5 if edge in edges_from + edges_to else 1 for
          edge in list(graph.edges)]

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    res_df.to_csv('Output/res2.csv', index=False)

    fig, ax = ox.plot_graph(graph, node_color=nc, node_size=ns, edge_color=ec, edge_linewidth=ew,
                            fig_height=18,
                            save=True, filename='res2')

    return node_num, oids_costs_from, oids_costs_to, oids_costs_sum


def get_minmax_obj_path(graph, graph_df, graph_edges, objects_list, nodes_list, output_path=config.path):
    res_dict = {}

    res_df = pd.DataFrame(columns=['src', 'dst', 'cost', 'path'])

    from_dict = {}
    for oid in objects_list:
        from_dict[oid] = dijkstra(graph_df, graph_edges, oid, set(nodes_list))

    temp_dict = {}

    for k, v in from_dict.items():

        max_cost = 0
        nnum = 0
        oid = 0
        for nid in nodes_list:
            try:
                cost = v[nid]['cost']
            except:
                print('.')
                print('k: ', k)
                print('v: ', v[nid])
                print('.')
            if cost > max_cost and cost != np.inf:
                max_cost = cost
                oid = k
            if oid != 0:
                temp_dict[objects_list.index(oid)] = (nodes_list.index(nid), max_cost, nid)

    onum_from = 0
    nnum_from = 0
    minmax_from = sys.maxsize
    for k, v in temp_dict.items():
        if v[1] < minmax_from:
            onum_from = k
            minmax_from = v[1]
            nnum_from = v[0]
            # nid_from = v[2]

    res_dict['from'] = (onum_from, nnum_from, minmax_from)

    way_from = from_dict[objects_list[onum_from]][nodes_list[nnum_from]]['path']
    cost_from = from_dict[objects_list[onum_from]][nodes_list[nnum_from]]['cost']
    subg_from = graph.subgraph(way_from)
    edges_from = subg_from.edges

    row_from = {'src': onum_from, 'dst': nnum_from, 'cost': cost_from, 'path': way_from}
    res_df = res_df.append(row_from, ignore_index=True)

    to_dict = {}
    for nid in nodes_list:
        to_dict[nid] = dijkstra(graph_df, graph_edges, nid, set(objects_list))

    temp_dict = {}
    for k, v in to_dict.items():
        max_cost = 0
        nnum = 0
        nid = 0
        for oid in objects_list:
            cost = v[oid]['cost']
            if cost > max_cost and cost != np.inf:
                max_cost = cost
                nid = k
            if nid != 0:
                temp_dict[objects_list.index(oid)] = (nodes_list.index(nid), max_cost, nid)

    onum_to = 0
    nnum_to = 0
    minmax_to = sys.maxsize
    for k, v in temp_dict.items():
        if v[1] < minmax_to:
            onum_to = k
            minmax_to = v[1]
            nnum_to = v[0]
            nid_to = v[2]

    res_dict['to'] = (onum_to, nnum_to, minmax_to)

    way_to = to_dict[nodes_list[nnum_to]][objects_list[onum_to]]['path']
    cost_to = to_dict[nodes_list[nnum_to]][objects_list[onum_to]]['cost']
    subg_to = graph.subgraph(way_to)
    edges_to = subg_to.edges

    row_to = {'src': onum_to, 'dst': nnum_to, 'cost': cost_to, 'path': way_to}
    res_df = res_df.append(row_to, ignore_index=True)

    temp_dict = {}
    for oid in objects_list:
        max_cost = 0
        nnum = 0
        nid_g = 0
        oid_g = 0
        for nid in nodes_list:
            cost_to = to_dict[nid][oid]['cost']
            cost_from = from_dict[oid][nid]['cost']
            if cost_from != np.inf and cost_to != np.inf:
                cost_sum = cost_from + cost_to
            if cost_sum > max_cost:
                max_cost = cost_sum
                nid_g = nid
                oid_g = oid
            if nid_g != 0 and oid_g != 0:
                temp_dict[objects_list.index(oid_g)] = (nodes_list.index(nid_g), max_cost, nid_g)

    onum_sum = 0
    nnum_sum = 0
    minmax_sum = sys.maxsize
    for k, v in temp_dict.items():
        if v[1] < minmax_sum:
            onum_sum = k
            minmax_sum = v[1]
            nnum_sum = v[0]
            nid_sum = v[2]
    res_dict['sum'] = (onum_sum, nnum_sum, minmax_sum)

    way_sum_to = to_dict[nodes_list[nnum_sum]][objects_list[onum_sum]]['path']
    cost_sum_to = to_dict[nodes_list[nnum_sum]][objects_list[onum_sum]]['cost']
    way_sum_from = from_dict[objects_list[onum_sum]][nodes_list[nnum_sum]]['path']
    cost_sum_from = from_dict[objects_list[onum_sum]][nodes_list[nnum_sum]]['cost']

    subg_sum_to = graph.subgraph(way_sum_to)
    subg_sum_from = graph.subgraph(way_sum_from)

    edges_sum_to = subg_sum_to.edges
    edges_sum_from = subg_sum_from.edges

    row_sum_from = {'src': onum_sum, 'dst': nnum_sum, 'cost': cost_sum_from, 'path': way_sum_from}
    res_df = res_df.append(row_sum_from, ignore_index=True)

    row_sum_to = {'src': nnum_sum, 'dst': onum_sum, 'cost': cost_sum_to, 'path': way_sum_to}
    res_df = res_df.append(row_sum_to, ignore_index=True)

    res_objects = [objects_list[onum_from], objects_list[onum_to], objects_list[onum_sum]]
    res_nodes = [nodes_list[nnum_from], nodes_list[nnum_to], nodes_list[nnum_sum]]
    res_edges = list(edges_from) + list(edges_to) + list(edges_sum_to) + list(edges_sum_from)
    sum_edges = list(edges_sum_to) + list(edges_sum_from)

    nc = ['r' if osmid in res_objects else 'g' if osmid in res_nodes else 'gray' for osmid in graph.nodes()]
    ns = [400 if osmid in res_objects or osmid in res_nodes else 10 for osmid in graph.nodes()]
    ec = ['orange' if edge in edges_from else 'yellow' if edge in edges_to else 'purple' if edge in sum_edges
    else 'gray' for edge in list(graph.edges)]
    ew = [5 if edge in res_edges else 1 for
          edge in list(graph.edges)]

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    res_df.to_csv('Output/res3.csv')

    fig, ax = ox.plot_graph(graph, node_color=nc, node_size=ns, edge_color=ec, edge_linewidth=ew,
                            fig_height=18,
                            save=True, filename='res3')

    return res_dict


def get_shortest_paths_sum(graph, graph_df, graph_edges, objects_list, nodes_list, output_path= config.path):
    distances = {}
    for oid in objects_list:
        distances[oid] = dijkstra(graph_df, graph_edges, oid, set(nodes_list))

    min_sum = sys.maxsize
    res_obj = 0
    for k, v in distances.items():
        obj_sum = 0
        for nid in nodes_list:
            if v[nid]['cost'] != np.inf:
                obj_sum += v[nid]['cost']
        if obj_sum < min_sum:
            min_sum = obj_sum
            res_obj = k

    res_obj_num = objects_list.index(res_obj)

    paths = []

    for nid in nodes_list:
        if distances[res_obj][nid]['cost'] != np.inf:
            paths.append(distances[res_obj][nid]['path'])

    edges = []
    for path in paths:
        path_subg = graph.subgraph(path)
        path_edges = path_subg.edges
        edges.append(path_edges)

    edges = [item for sublist in edges for item in sublist]

    res_dict = {}
    for k, v in distances[res_obj].items():
        res_dict[nodes_list.index(k)] = v

    res_df = pd.DataFrame(res_dict).T

    nc = ['r' if osmid == res_obj else 'g' if osmid in nodes_list else 'gray' for osmid in graph.nodes()]
    ns = [400 if osmid == res_obj or osmid in nodes_list else 10 for osmid in graph.nodes()]
    ec = ['orange' if edge in edges else 'gray' for edge in list(graph.edges)]
    ew = [5 if edge in edges else 1 for edge in list(graph.edges)]

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    res_df.to_csv('Output/res4.csv'.format(res_obj_num))

    fig, ax = ox.plot_graph(graph, node_color=nc, node_size=ns, edge_color=ec, edge_linewidth=ew,
                            fig_height=18,
                            save=True, filename='res4')

    return res_obj_num, min_sum


def get_lightest_tree(graph, graph_df, graph_edges, objects_list, nodes_list, output_path= config.path):
    distances = {}
    for oid in objects_list:
        distances[oid] = dijkstra(graph_df, graph_edges, oid, set(nodes_list))

    res = {}

    for k, v in distances.items():
        tree_edges = []
        dist = 0
        for nid in nodes_list:
            curr_path = v[nid]['path']
            for i in range(1, len(curr_path)):
                root = curr_path[i - 1]
                dst = curr_path[i]
                edge = (root, dst)
                if edge not in tree_edges:
                    tree_edges.append(edge)
                    x1 = graph_df.loc[root, 'x']
                    y1 = graph_df.loc[root, 'y']
                    x2 = graph_df.loc[dst, 'x']
                    y2 = graph_df.loc[dst, 'y']
                    dist += converting_to_csv.gaversin_distance(x1, y1, x2, y2)

        res[k] = dist

        res_obj = 0
        min_weight = sys.maxsize
        for k, v in res.items():
            if v < min_weight:
                res_obj = k
                min_weight = v

    res_obj_num = objects_list.index(res_obj)

    paths = []

    for nid in nodes_list:
        if distances[res_obj][nid]['cost'] != np.inf:
            paths.append(distances[res_obj][nid]['path'])

    edges = []
    for path in paths:
        path_subg = graph.subgraph(path)
        path_edges = path_subg.edges
        edges.append(path_edges)

    res_dict = {}
    for k, v in distances[res_obj].items():
        res_dict[nodes_list.index(k)] = v

    res_df = pd.DataFrame(res_dict).T

    edges = [item for sublist in edges for item in sublist]

    nc = ['r' if osmid == res_obj else 'g' if osmid in nodes_list else 'gray' for osmid in graph.nodes()]
    ns = [400 if osmid == res_obj or osmid in nodes_list else 10 for osmid in graph.nodes()]
    ec = ['orange' if edge in edges else 'gray' for edge in list(graph.edges)]
    ew = [5 if edge in edges else 1 for edge in list(graph.edges)]

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    res_df.to_csv('Output/res5.csv'.format(res_obj_num))

    fig, ax = ox.plot_graph(graph, node_color=nc, node_size=ns, edge_color=ec, edge_linewidth=ew,
                            fig_height=18,
                            save=True, filename='res5')

    return res_obj_num, min_weight