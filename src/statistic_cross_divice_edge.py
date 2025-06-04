def load_nodes_from_file(node_file_path):
    nodes = []
    with open(node_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(int(line))
    return nodes

def load_device_map(device_nodes_files):
    device_map = {}
    for device_id, file_path in enumerate(device_nodes_files):
        nodes = load_nodes_from_file(file_path)
        for node in nodes:
            device_map[node] = device_id
    return device_map

def count_cross_device_edges(edge_file_path, device_map):
    cross_count = 0
    total_edges = 0
    with open(edge_file_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            dev_u = device_map.get(u, -1)
            dev_v = device_map.get(v, -1)
            if dev_u == -1 or dev_v == -1:
                print(f"Warning: node {u} or {v} not found in device map.")
                continue
            total_edges += 1
            if dev_u != dev_v:
                cross_count += 1
    return cross_count, total_edges

if __name__ == "__main__":
    device_files = [
        'res/extrem_cur_nodes_0.txt',
        'res/extrem_cur_nodes_1.txt',
        'res/extrem_cur_nodes_2.txt',
        'res/extrem_cur_nodes_3.txt'
    ]

    edge_file = 'data/maxcut_data/stanford_data_single/G22.txt'

    device_map = load_device_map(device_files)
    cross_edges, total_edges = count_cross_device_edges(edge_file, device_map)

    print(f"Total edges: {total_edges}")
    print(f"Cross-device edges: {cross_edges}")
