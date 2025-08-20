from network.detector import Detector,MultDetector
from network.refiner import VolumeRefiner
from network.selector import ViewpointSelector

name2network={
    'refiner': VolumeRefiner,
    'detector': Detector,
    'mult_detector':MultDetector,
    'selector': ViewpointSelector,
}