import random
import numpy as np


TRACKS = [
    ("e-track-1", "road"),
    ("e-track-2", "road"),
    ("e-track-3", "road"),
    ("e-track-4", "road"),
    ("e-track-6", "road"),
]

NEW_TRACKS = [
    ("e-track-1", "road"),
    ("e-track-2", "road"),
    ("e-track-3", "road"),
    ("e-track-4", "road"),
    ("e-track-6", "road"),
    ("eroad", "road"),
    ("forza", "road"),
    ("g-track-1", "road"),
    ("g-track-2", "road"),
    ("g-track-3", "road"),
    ("ole-road-1", "road"),
    ("ruudskogen", "road"),
    ("street-1", "road"),
    ("wheel-1", "road"),
    ("wheel-2", "road"),
    ("aalborg", "road"),
    ("alpine-1", "road"),
    ("alpine-2", "road"),
]


def _find_by_name(nodes, key):
    for node in nodes:
        if node.attrib["name"] == key:
            return node


def _find_by_tag(nodes, tag):
    for node in nodes:
        if node.tag == tag:
            return node


def sample_track(root_node, counter, track):
    node = _find_by_name(root_node, "Tracks")
    subnode = _find_by_tag(node, "section")
    trackname_node, tracktype_node = subnode.getchildren()
    
    # trackname, tracktype = random.sample(TRACKS, 1)[0]
    # trackname, tracktype = random.sample(NEW_TRACKS, 1)[0]
    if track != 'none':
        trackname, tracktype = track, 'road'
    else:
        trackname, tracktype = NEW_TRACKS[counter % len(NEW_TRACKS)]
    print(trackname, tracktype)

    trackname_node.attrib["val"] = trackname
    tracktype_node.attrib["val"] = tracktype

    return trackname, tracktype


def set_render_mode(root_node, render=True):
    node = _find_by_name(root_node, "Quick Race")
    subnode = _find_by_name(node, "display mode")
    if render:
        subnode.attrib["val"] = "normal"
    else:
        subnode.attrib["val"] = "results only"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
