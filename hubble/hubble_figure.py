import reproject
from astropy.io import fits
from astropy import wcs
import pylab as pl
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['image.interpolation'] = 'nearest'

fh = fits.open('hst_paa_smoothed_to_2arcsec_with_simulated_noise.fits')
fh2 = fits.open('gc_mosaic_miris_line_minus_cont_scaled_pow1.1_x0p35.fits')

target_header = {'CRPIX1': 1024, 'CRPIX2': 1024, 'CRVAL1': 0.00478, 'CRVAL2':-0.00561, 'CDELT1':-0.942/3600, 'CDELT2': 0.942/3600, 'CTYPE1':'GLON-CAR', 'CTYPE2':'GLAT-CAR'}
target_wcs = wcs.WCS(target_header)

hst_repr, _ = reproject.reproject_interp(fh, target_wcs, shape_out=(2048,2048))
miris_repr, _ = reproject.reproject_exact(fh2, target_wcs, shape_out=(2048,2048))

fig = pl.figure(1)
fig.clf()
fig.add_axes(projection=target_wcs)
ax = fig.gca()
ax.imshow(miris_repr, interpolation='none', cmap='gray_r')
ax.imshow(hst_repr, cmap='gray_r', vmin=0, vmax=10000)
ax.add_patch(pl.Rectangle((0,0), 2048, 500, fill=False, color='r', linewidth=5))
ax.add_patch(pl.Rectangle((0,2048-500), 2048, 500, fill=False, color='r', linewidth=5))
ax.add_patch(pl.Rectangle((0,524), 2048, 1000, fill=False, color='b', linewidth=5))
ax.axis([-2,2050,-2,2050])

pl.savefig('../paper/figures/hubble_miris_paa_with_overlay.pdf', bbox_inches='tight', dpi=250)
