import math
from shapely.geometry.polygon import Polygon
import numpy as np


class Partition:
    def __init__(self):
        pass

    partitionId = 0
    numRecords = 0
    filesize = 0
    filename = ""
    x1, y1, x2, y2 = 0, 0, 0, 0
    mbr = Polygon()
    nblocks = 1


def get_partitions(filename, block_size):

    partitions = []
    with open(filename) as f:
        lines = f.readlines()
        lines.pop(0)
        for line in lines:
            values = line.split('\t')
            partition = Partition()
            try:
                # partition.partitionId = int(values[0])
                partition.numRecords = int(values[2])
                partition.filesize = int(values[3])
                partition.filename = values[1].replace("'", "")
                partition.x1, partition.y1, partition.x2, partition.y2 = float(values[5]), float(values[6]), float(values[7]), float(values[8])
                partition.nblocks = math.ceil(float(values[3]) / (block_size * 1024 * 1024))
                partition.mbr = Polygon([(partition.x1, partition.y1), (partition.x1, partition.y2), (partition.x2, partition.y2), (partition.x2, partition.y1)])
            except Exception as e:
                print(filename)
                print(e)
            partitions.append(partition)
    return partitions


def combine_partitions(partitions, block_size):
    partition = Partition()
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for p in partitions:
        partition.numRecords += p.numRecords
        partition.filesize += p.filesize
        if p.x1 < min_x:
            min_x = p.x1
        if p.y1 < min_y:
            min_y = p.y1
        if p.x2 > max_x:
            max_x = p.x2
        if p.y2 > max_y:
            max_y = p.y2

    partition.x1, partition.y1, partition.x2, partition.y2 = min_x, min_y, max_x, max_y
    partition.mbr = Polygon([(partition.x1, partition.y1), (partition.x1, partition.y2), (partition.x2, partition.y2), (partition.x2, partition.y1)])
    partition.nblocks = math.ceil(float(partition.filesize) / (block_size * 1024 * 1024))

    return partition


def get_total_area(partitions):
    total_area = 0
    for p in partitions:
        total_area += abs(p.x2 - p.x1) * abs(p.y2 - p.y1) * p.nblocks
    return total_area


def get_total_margin(partitions):
    total_margin = 0
    for p in partitions:
        total_margin += (abs(p.x2 - p.x1) + abs(p.y2 - p.y1)) * p.nblocks
    return total_margin


def get_total_overlap(partitions):
    total_overlap = 0
    for p1 in partitions:
        total_overlap += p1.mbr.area * p1.nblocks * (p1.nblocks - 1) / 2
        for p2 in partitions:
            if p1 is not p2 and p1.mbr.intersects(p2.mbr):
                total_overlap += p1.mbr.intersection(p2.mbr).area * p1.nblocks * p2.nblocks

    return total_overlap


def get_size_std(partitions):
    sizes = []
    for p in partitions:
        sizes.append(p.filesize)
    size_arr = np.array(sizes)
    size_std = np.std(size_arr)
    return size_std


def get_disk_util(partitions, block_size):
    value = 0
    count = 0
    for p in partitions:
        p_util = p.filesize / (float) (block_size * 1024 * 1024)
        value += p_util
        count += p.nblocks
    util = value / count
    return util


def get_total_blocks(partitions):
    total_blocks = 0
    for p in partitions:
        total_blocks += p.nblocks
    return total_blocks


def get_cost(partitions, query_ratio, min_x, min_y, max_x, max_y):
    cost = 0
    w = max_x - min_x
    h = max_y - min_y
    q = math.sqrt(query_ratio * w * h)
    for p in partitions:
        cost += (abs(p.x2 - p.x1) + q) * (abs(p.y2 - p.y1) + q) * p.nblocks / (w * h)
    return cost
