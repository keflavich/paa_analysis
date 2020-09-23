import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy import convolution
from matplotlib import patheffects
from reproject import reproject_from_healpix, reproject_to_healpix
from astropy import time, coordinates
from astropy import units as u

import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
pl.ion()

# Use the ROSAT image to get an all-sky header
filename = get_pkg_data_filename('allsky/allsky_rosat.fits')
hdu = fits.open(filename)[0]
wcs = WCS(hdu.header)

hihdu = fits.open('https://lambda.gsfc.nasa.gov/data/foregrounds/ebv_2017/mom0_-90_90_1024.hpx.fits')
array, footprint = reproject_from_healpix(hihdu[1], hdu.header)

fig = pl.figure(1, figsize=(20,15))
fig.clf()
ax = plt.subplot(projection=wcs, frame_class=EllipticalFrame)

path_effects=[patheffects.withStroke(linewidth=3, foreground='black')]
#ax.coords.grid(color='black')
#ax.coords['glon'].set_ticklabel(color='cyan', path_effects=path_effects)
ax.coords['glon'].set_ticks_visible(False)
ax.coords['glat'].set_ticks_visible(False)
ax.coords['glon'].set_ticklabel_visible(False)
ax.coords['glat'].set_ticklabel_visible(False)
transform = ax.get_transform('galactic')

im = ax.imshow(array, origin='lower', cmap='gray_r', interpolation='none')

# Clip the image to the frame
im.set_clip_path(ax.coords.frame.patch)


# plot our pointing on the map
times = time.Time(np.linspace(2025.0, 2025.999, 365*24), format='decimalyear')
times = time.Time(np.linspace(2025 + 21/365, 2025+355/365, 12), format='decimalyear')
#observer = coordinates.get_body('Earth', times)
observer = coordinates.EarthLocation.from_geodetic(-82.3*u.deg, 29.6*u.deg, 600*u.km)
sun = coordinates.get_body(body='Sun', time=times)
# MUST drop the time here: conversion to Galactic coords does stupid things otherwise
sun = coordinates.SkyCoord(sun.spherical, frame='fk5').galactic

sort_ell = np.argsort(sun.l)

ell, bee = sun.l[sort_ell], sun.b[sort_ell]

# green is the full track, red is the last obs
#ax.plot(ell[ell<180*u.deg], bee[ell<180*u.deg], color='r', transform=transform)
#ax.plot(ell[ell>180*u.deg], bee[ell>180*u.deg], color='r', transform=transform)
#sc=ax.scatter(ell[ok], bee[ok], c=velo_measurement[ok].value, transform=transform, cmap=pl.cm.RdBu_r)
#pl.colorbar(mappable=sc)
#ax.scatter(target.l.deg, target.b.deg, color='g', s=100, transform=transform)

axlims = pl.axis()

ellgrid,beegrid = np.mgrid[-180:180,-90:90]
coordgrid = coordinates.SkyCoord(ellgrid*u.deg, beegrid*u.deg, frame='galactic')

lmcfh = fits.open('../lmc/LMC_HI_ATCA_peak.fits')
smlmc = convolution.convolve_fft(lmcfh[0].data.squeeze(), convolution.Gaussian2DKernel(25))
lmcwcs = WCS(lmcfh[0].header)
lmc_transform = ax.get_transform(lmcwcs.celestial)
ax.contourf(smlmc, levels=[13, 30, 1e4], transform=lmc_transform, colors=[(1,165./256,0,0.5), (1,165./256,0,0.75), (1,165./256,0,1)])

#lmc = coordinates.SkyCoord.from_name('LMC').galactic
#ax.scatter(lmc.l, lmc.b, transform=transform, s=500, marker='o', facecolor=(1,0.5,0,0.3), edgecolor='none')


ngc6503 = coordinates.SkyCoord.from_name('ngc6503').galactic
ax.scatter(ngc6503.l, ngc6503.b, transform=transform, s=500, marker='o', facecolor=(1,0,0.5,0.3), edgecolor='none')

# m33 = coordinates.SkyCoord.from_name('M33').galactic
# ax.scatter(m33.l, m33.b, transform=transform, s=100, marker='s', facecolor=(0,1,0.1,0.4), edgecolor='none')
# wlm = coordinates.SkyCoord.from_name('WLM Galaxy').galactic
# ax.scatter(wlm.l, wlm.b, transform=transform, s=100, marker='s', facecolor=(0,0.1,1,0.4), edgecolor='none')
# ngc6822 = coordinates.SkyCoord.from_name('NGC 6822').galactic
# ax.scatter(ngc6822.l, ngc6822.b, transform=transform, s=100, marker='s', facecolor=(1,0.1,0,0.4), edgecolor='none')

ax.plot([50,50,2.5,2.5,-2.5,-2.5,-50,-50, -2.5, -2.5, 2.5, 2.5, 50], [1, -1, -1, -5, -5, -1, -1, 1, 1, 5, 5, 1, 1], transform=transform, color='r')

months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

for thistime,sunpos in zip(times, sun):
    sep = coordgrid.separation(sunpos)
    ell, bee = sunpos.l, sunpos.b
    sc = ax.scatter(ell, bee, marker='*', c='yellow', transform=transform, edgecolor='k', s=200)
    #ax.pcolormesh(ellgrid, beegrid, (sep>85*u.deg) & (sep < 135*u.deg), transform=transform)
    con = ax.contourf(ellgrid, beegrid, (sep>85*u.deg) & (sep < 135*u.deg), levels=[0.5, 1.5], transform=transform, alpha=0.25)
    #ax.contour(ellgrid, beegrid, (sep>85*u.deg) & (sep < 135*u.deg), levels=[0.5, 1.5], transform=transform)

    month = months[thistime.ymdhms.month-1]
    print(month, thistime, thistime.ymdhms)
    #ax.set_title(month)
    txt = ax.text(0.2, 0.7, month, transform=fig.transFigure)
    pl.axis(axlims)
    pl.draw()
    pl.show()
    pl.pause(0.01)
    fig.savefig(f'figures/observable_zone_{month}.png', bbox_inches='tight')
    txt.set_visible(False)

    sc.set_visible(False)
    for cc in con.collections:
        cc.remove()
