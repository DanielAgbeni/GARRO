"""
GARRO Topology Package.

Available topology builders:
    get_nsfnet()       → 14-node US WAN backbone
    get_geant2()       → 24-node European academic network
    get_fat_tree(k=8)  → k-ary Fat-Tree data centre topology
"""
from topologies.nsfnet import get_nsfnet
from topologies.geant2 import get_geant2
from topologies.fat_tree import get_fat_tree

__all__ = ["get_nsfnet", "get_geant2", "get_fat_tree"]
