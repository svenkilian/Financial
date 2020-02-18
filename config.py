"""
This module serves to provide the project's root path through the ROOT_DIR variable
"""

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# Update matplotlib settings
plt.rcParams.update({'legend.fontsize': 8,
                     'legend.loc': 'best',
                     'legend.markerscale': 2.5,
                     'legend.frameon': True,
                     'legend.fancybox': True,
                     'legend.shadow': True,
                     'legend.facecolor': 'w',
                     'legend.edgecolor': 'black',
                     'legend.framealpha': 1})

# matplotlib.style.use('fivethirtyeight')
plt_styles = ['seaborn-ticks', 'ggplot', 'dark_background', 'bmh', 'seaborn-poster', 'seaborn-notebook', 'fast',
              'seaborn',
              'classic', 'Solarize_Light2', 'seaborn-dark', 'seaborn-pastel', 'seaborn-muted', '_classic_test',
              'seaborn-paper',
              'seaborn-colorblind', 'seaborn-bright', 'seaborn-talk', 'seaborn-dark-palette', 'tableau-colorblind10',
              'seaborn-darkgrid', 'seaborn-whitegrid', 'fivethirtyeight', 'grayscale', 'seaborn-white', 'seaborn-deep']

matplotlib.style.use('ggplot')
register_matplotlib_converters()

pd.set_option('precision', 6)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 40)
pd.set_option('max_colwidth', 80)
pd.set_option('mode.sim_interactive', True)
pd.set_option('expand_frame_repr', True)
pd.set_option('large_repr', 'truncate')

pd.set_option('colheader_justify', 'left')
pd.set_option('display.width', 800)
pd.set_option('display.html.table_schema', False)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
run_id = None
study_period_id = None
