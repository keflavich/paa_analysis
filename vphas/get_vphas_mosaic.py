import numpy as np
import os
from astropy import constants, units as u, table, stats, coordinates, wcs, log, coordinates as coord, convolution, modeling, time; from astropy.io import fits, ascii
import reproject
import glob
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd
from reproject import reproject_interp

from reproject.utils import parse_input_data, parse_input_weights, parse_output_projection
from reproject.mosaicking.background import determine_offset_matrix, solve_corrections_sgd
from reproject.mosaicking.subset_array import ReprojectedArraySubset


import requests
from bs4 import BeautifulSoup

from astropy.utils.console import ProgressBar

from astroquery.eso import Eso

files = glob.glob("*ADP*fits*")
if len(files) < 732:
    Eso.ROW_LIMIT = 10000
    Eso.cache_location = '.'
    Eso.login()

    tbl = Eso.query_surveys(surveys='VPHASplus', coord1=0, coord2=0, coord_sys='gal', box=5*u.deg)
    files = Eso.retrieve_data(tbl['ARCFILE'])

hdus = [hdu
        for h in (fits.open(x)
                  for x in glob.glob("ADP*.fits*"))
        for hdu in h if 'CRVAL1' in hdu.header and (260 < hdu.header['CRVAL1'] < 270)]
#wcs_out, shape_out = find_optimal_celestial_wcs([h[1] for h in hdus], frame='galactic')

wcs_out = wcs.WCS({"CTYPE1": "GLON-CAR",
                   "CTYPE2": "GLAT-CAR",
                   "CRPIX1": 14000.0,
                   "CRPIX2": 14000.0,
                   "CRVAL1": 0.0,
                   "CRVAL2": 0.0,
                   "CDELT1": -1/3600., # downsample to 1" pixels
                   "CDELT2": 1/3600.,
                   "CUNIT1": "deg",
                   "CUNIT2": "deg",
                  })
shape_out = (28000,28000)

# [hdu for h in hdus for hdu in h[1:]]
#array_line, footprint = reproject_and_coadd(hdus, wcs_out, shape_out=shape_out, reproject_function=reproject_interp)
#fits.PrimaryHDU(data=array_line, header=wcs_out.to_header()).writeto('gc_vphas_mosaic_halpha.fits', overwrite=True)

header = wcs_out.to_header()
header['BITPIX'] = -32
header['NAXIS1'] = shape_out[0]
header['NAXIS2'] = shape_out[0]
if not os.path.exists('gc_vphas_mosaic_halpha.fits'):
    header.tofile('gc_vphas_mosaic_halpha.fits')
if not os.path.exists('gc_vphas_mosaic_halpha_coverage.fits'):
    header.tofile('gc_vphas_mosaic_halpha_coverage.fits')

output_file = fits.open('gc_vphas_mosaic_halpha.fits', mode='update')
if output_file[0].data.shape != shape_out:
    output_file.close()
    with open('gc_vphas_mosaic_halpha.fits', 'rb+') as fobj:
        fobj.seek(len(header.tostring()) + (shape_out[0] * shape_out[1] * 4) - 1)
        fobj.write(b'\0')
    output_file = fits.open('gc_vphas_mosaic_halpha.fits', mode='update')

output_coverage = fits.open('gc_vphas_mosaic_coverage.fits', mode='update')
if output_coverage[0].data.shape != shape_out:
    output_coverage.close()
    with open('gc_vphas_mosaic_halpha_coverage.fits', 'rb+') as fobj:
        fobj.seek(len(header.tostring()) + (shape_out[0] * shape_out[1] * 4) - 1)
        fobj.write(b'\0')
    output_coverage = fits.open('gc_vphas_mosaic_coverage.fits', mode='update')


final_array = output_file[0].data
final_footprint = output_coverage[0].data

input_data = hdus
output_projection = wcs_out

# input parameters we're not using
hdu_in = None
input_weights = None # might want this to be output_coverage?
hdu_weights = None
reproject_function = reproject_interp
kwargs = {}
match_background = False
background_reference = None
combine_function = 'mean'

# Parse the output projection to avoid having to do it for each
wcs_out, shape_out = parse_output_projection(output_projection,
                                             shape_out=shape_out)

# Start off by reprojecting individual images to the final projection

arrays = []

for idata in ProgressBar(range(len(input_data))):

    # We need to pre-parse the data here since we need to figure out how to
    # optimize/minimize the size of each output tile (see below).
    array_in, wcs_in = parse_input_data(input_data[idata], hdu_in=hdu_in)

    # We also get the weights map, if specified
    if input_weights is None:
        weights_in = None
    else:
        weights_in = parse_input_weights(input_weights[idata], hdu_weights=hdu_weights)
        if np.any(np.isnan(weights_in)):
            weights_in = np.nan_to_num(weights_in)

    # Since we might be reprojecting small images into a large mosaic we
    # want to make sure that for each image we reproject to an array with
    # minimal footprint. We therefore find the pixel coordinates of corners
    # in the initial image and transform this to pixel coordinates in the
    # final image to figure out the final WCS and shape to reproject to for
    # each tile. Note that in future if we are worried about significant
    # distortions of the edges in the reprojection process we could simply
    # add arbitrary numbers of midpoints to this list.
    ny, nx = array_in.shape
    xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    xc_out, yc_out = wcs_out.world_to_pixel(wcs_in.pixel_to_world(xc, yc))

    # Determine the cutout parameters

    # In some cases, images might not have valid coordinates in the corners,
    # such as all-sky images or full solar disk views. In this case we skip
    # this step and just use the full output WCS for reprojection.

    if np.any(np.isnan(xc_out)) or np.any(np.isnan(yc_out)):
        imin = 0
        imax = shape_out[1]
        jmin = 0
        jmax = shape_out[0]
    else:
        imin = max(0, int(np.floor(xc_out.min() + 0.5)))
        imax = min(shape_out[1], int(np.ceil(xc_out.max() + 0.5)))
        jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
        jmax = min(shape_out[0], int(np.ceil(yc_out.max() + 0.5)))

    if imax < imin or jmax < jmin:
        continue

    wcs_out_indiv = wcs_out[jmin:jmax, imin:imax]
    shape_out_indiv = (jmax - jmin, imax - imin)

    # TODO: optimize handling of weights by making reprojection functions
    # able to handle weights, and make the footprint become the combined
    # footprint + weight map

    array, footprint = reproject_function((array_in, wcs_in),
                                          output_projection=wcs_out_indiv,
                                          shape_out=shape_out_indiv,
                                          hdu_in=hdu_in,
                                          **kwargs)

    if weights_in is not None:
        weights, _ = reproject_function((weights_in, wcs_in),
                                        output_projection=wcs_out_indiv,
                                        shape_out=shape_out_indiv,
                                        hdu_in=hdu_in,
                                        **kwargs)

    # For the purposes of mosaicking, we mask out NaN values from the array
    # and set the footprint to 0 at these locations.
    reset = np.isnan(array)
    array[reset] = 0.
    footprint[reset] = 0.

    # Combine weights and footprint
    if weights_in is not None:
        weights[reset] = 0.
        footprint *= weights

    array = ReprojectedArraySubset(array, footprint,
                                   imin, imax, jmin, jmax)

    array.array[array.footprint == 0] = 0
    final_array[array.view_in_original_array] += array.array * array.footprint
    final_footprint[array.view_in_original_array] += array.footprint
    del array
    output_file.flush()
    output_coverage.flush()

final_array /= final_footprint
output_file.close()
