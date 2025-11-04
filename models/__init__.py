import sys,os
dir=os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir)
from FFA import FFA,FFA1, FFA2, FFA3, FFA4
from PerceptualLoss import LossNetwork as PerLoss
