import numpy as np
from astropy import constants, units as u, table, stats, coordinates, wcs, log, coordinates as coord, convolution, modeling, time; from astropy.io import fits, ascii
import radio_beam
from pyspeckit.spectrum.models import hydrogen

fh = fits.open('hlsp_hpsgc_hst_nicmos-nic3_gc_palpha_v1_img.fits')
ww = wcs.WCS(fh[0].header)
beam = radio_beam.Beam((1.8756*u.um / (2.4*u.m)).to(u.arcsec, u.dimensionless_angles()))
target_beam = radio_beam.Beam(2*u.arcsec)
kernel = target_beam.deconvolve(beam)

pixscale = wcs.utils.proj_plane_pixel_scales(ww).mean() * u.deg
sm = convolution.convolve(fh[0].data, kernel.as_kernel(pixscale))

fh[0].data = sm[::10, ::10]
fh[0].header.update(ww[::10,::10].to_header())

fh.writeto("hst_paa_smoothed_to_2arcsec.fits")

in_unit = u.uJy/(pixscale**2)
bandwidth = 187*u.nm
wl_paa = hydrogen.wavelength['paschena']*u.um
e_paa = wl_paa.to(u.erg, u.spectral())
nu_paa = wl_paa.to(u.Hz, u.spectral())
sb_unit = u.erg/u.s/u.cm**2/u.arcsec**2

conversion_factor = (in_unit*(bandwidth / wl_paa) * nu_paa).to(sb_unit)

throughput_new = 0.37
throughput = throughput_new

sb_sensitivity = 1.67e-16*sb_unit
aperture_diameter = 24*u.cm

collecting_area = (aperture_diameter/2)**2 * np.pi
effective_area = collecting_area*throughput
fiducial_integration_time = 500*u.s

pixscale_new = pixscale * 10
counts = sm[::10, ::10] * (conversion_factor * effective_area * fiducial_integration_time / e_paa * pixscale_new**2).decompose()
counts[sm[::10,::10]==0] = np.nan

readnoise = 6.2
counts += 0.123/u.s * fiducial_integration_time

sim_image = counts + np.random.randn(*counts.shape)*counts.value**0.5 + np.random.randn(*counts.shape)*readnoise
fh[0].data = sim_image.value
fh.writeto("hst_paa_smoothed_to_2arcsec_with_simulated_noise.fits", overwrite=True)
