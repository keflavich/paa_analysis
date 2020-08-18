# This script was produced by glue and can be used to further customize a
# particular plot.

### Package imports

from glue.core.state import load
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
#matplotlib.use('Agg')

### Set up data

data_collection = load('make_plot.py.data')

### Set up viewer

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect='auto')

### Set up layers

## Layer 1: coelho14_model_paa[HDU1]

layer_data = data_collection[0]

# Get main data values
x = layer_data['mag_paa_m_high']
y = layer_data['mag_paa_m_low']
keep = ~np.isnan(x) & ~np.isnan(y)

ax.plot(x[keep], y[keep], 'o', color='#595959', markersize=3, alpha=0.8, zorder=1, mec='none')
## Layer 2: Subset 1

layer_data = data_collection[0].subsets[0]

# Get main data values
x = layer_data['mag_paa_m_high']
y = layer_data['mag_paa_m_low']
keep = ~np.isnan(x) & ~np.isnan(y)

ax.plot(x[keep], y[keep], 'o', color='#e31a1c', markersize=7, alpha=0.5, zorder=2, mec='none')
## Layer 3: Subset 2

## Layer 4: Subset 3

## Layer 5: Subset 4

## Layer 6: Subset 5

layer_data = data_collection[0].subsets[4]

# Get main data values
x = layer_data['mag_paa_m_high']
y = layer_data['mag_paa_m_low']
keep = ~np.isnan(x) & ~np.isnan(y)

ax.plot(x[keep], y[keep], 'o', color='#ff7f00', markersize=7, alpha=0.5, zorder=6, mec='none')

### Finalize viewer

# Set limits
ax.set_xlim(1.6696599130554277, 1.974161263538064)
ax.set_ylim(2.712930717573334, 2.9810489651222967)

# Set scale (log or linear)
ax.set_xscale('linear')
ax.set_yscale('linear')

# Set axis label properties
ax.set_xlabel(r'$M_{Pa\alpha} - M_{cont,high}$', weight='normal', size=15)
ax.set_ylabel(r'$M_{Pa\alpha} - M_{cont,low}$', weight='normal', size=15)

# Set tick label properties
ax.tick_params('x', labelsize=12)
ax.tick_params('y', labelsize=12)

# Save figure
fig.savefig('paa_cont_colorcolor.pdf', bbox_inches='tight')
#plt.close(fig)
