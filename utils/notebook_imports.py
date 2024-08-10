"""
    just paste a bunch of frequently used libs, so that I can import this one in my notebook
"""

import numpy as np
import torch
import torch.nn as nn
import nltk
import pandas as pd
from tqdm.notebook import tqdm as nbtqdm
from ipywidgets import interact
from IPython.display import display
from importlib import reload

import os
import sys
from os.path import join as joinpath
import json
from typing import Dict, List, Optional
import re
import time
from functools import partial
import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

from .misc import all_exists, any_exists, jdump, jload
