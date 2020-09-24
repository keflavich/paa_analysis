import numpy as np
from astropy import units as u
from pyspeckit.spectrum.models import hydrogen
from astropy import convolution
from astropy import modeling

# SPAM calcs
# sb_sensitivity is the anticipated surface brightness sensitivity

sb_unit = u.erg/u.s/u.cm**2/u.arcsec**2

throughput_old = 0.7
throughput_new = 0.37
#throughput_new = 0.7
throughput = throughput_new

sb_sensitivity = 1.5e-16 *u.erg/u.s/u.cm**2/u.arcsec**2 * (throughput_old/throughput_new) # throughput correction
aperture_diameter = 24*u.cm

collecting_area = (aperture_diameter/2)**2 * np.pi
effective_area = collecting_area*throughput

paa_bandwidth = 5*u.nm
paac_bandwidth = 10*u.nm

wl_paa = hydrogen.wavelength['paschena']*u.um
e_paa = wl_paa.to(u.erg, u.spectral())
nu_paa = wl_paa.to(u.Hz, u.spectral())

wl_halpha = hydrogen.wavelength['balmera']*u.um
nu_halpha = wl_halpha.to(u.Hz, u.spectral())
e_halpha = wl_halpha.to(u.eV, u.spectral())

paa_bandwidth_Hz = ((paa_bandwidth / wl_paa) * nu_paa).to(u.Hz)

sb_sensitivity_MJySr = (sb_sensitivity).to(u.MJy/u.sr, u.spectral_density(wl_paa))

pixscale = (0.806*u.arcsec)
pixscale = 0.942*u.arcsec
fov = pixscale * 2048

fwhm = (1.22*wl_paa / aperture_diameter).to(u.arcsec, u.dimensionless_angles())
psf_area = 2*np.pi*(fwhm**2) / (8*np.log(2))
ppbeam = psf_area / (pixscale**2)



readnoise_pessimistic = 22*u.count
readnoise_optimistic = 6.2*u.count
dark_rate_optimistic = 0.123*u.count/u.s
dark_rate_pessimistic = 0.435*u.count/u.s

fiducial_integration_time = 500*u.s


saturation_limit = 1.8e5

yy,xx = np.mgrid[-25:25:1,-25:25:1]
airymod = modeling.models.AiryDisk2D(amplitude=1, x_0=0, y_0=0, radius=fwhm/pixscale)(yy,xx)
airypeakfrac = (airymod.max() / airymod.sum())

max_unsaturated_rate = saturation_limit / fiducial_integration_time
max_unsaturated_sb = (e_paa / effective_area * max_unsaturated_rate / pixscale**2).to(sb_unit)
max_unsaturated_flux = (e_paa / effective_area / airypeakfrac * max_unsaturated_rate / paa_bandwidth_Hz).to(u.mJy)
mag_zeropoint_paa = 870*u.Jy # interpolated
max_unsaturated_mag = -2.5 * np.log10(max_unsaturated_flux / mag_zeropoint_paa)

# assume we can average down noise by sqrt(npixels) (slightly optimistic) to extract point sources
ps_sensitivity = sb_sensitivity * psf_area / ppbeam**0.5
sensitivity_mag = -2.5*np.log10(ps_sensitivity / (mag_zeropoint_paa*paa_bandwidth_Hz))

# this isn't right; the 1-pixel version needs to be _less_ sensitive.
#ps_sensitivity_1pix = sb_sensitivity * pixscale**2
#sensitivity_mag_1pix = -2.5*np.log10(ps_sensitivity_1pix/nu_paa / mag_zeropoint_paa)
#detectionlimit_mag_1pix = -2.5*np.log10(ps_sensitivity_1pix*5/nu_paa / mag_zeropoint_paa)

magzero_1pix_count_rate = (mag_zeropoint_paa * effective_area * paa_bandwidth_Hz / e_paa).decompose() * (airymod.max() / airymod.sum()) * u.count
# 5-sigma detection where sigma is readnoise only in 1/200th of a second
pointing_integration = 1/(50*u.Hz)
pointing_star_sn = ((magzero_1pix_count_rate * pointing_integration) / (readnoise_optimistic)).decompose()
snr_for_pointing = 5
magnitude_for_SN5 = -2.5*np.log10(snr_for_pointing/pointing_star_sn)


rn_pess = readnoise_pessimistic/fiducial_integration_time
# Poisson noise
darkn_pess = ((dark_rate_pessimistic * fiducial_integration_time)**0.5).value * u.count / fiducial_integration_time
dark_rn_pess = (((rn_pess + darkn_pess) * (e_paa/u.count) /
                 (collecting_area * throughput) / nu_paa).to(u.mJy) /
                pixscale**2).to(u.MJy/u.sr)

rn_opt = readnoise_optimistic.value
darkn_opt = ((dark_rate_optimistic * fiducial_integration_time)).value

dark_rn_opt = (((rn_opt**2 + darkn_opt)**0.5 / fiducial_integration_time * (e_paa) /
                effective_area) /
               pixscale**2)
calculated_sb_sens = dark_rn_opt.to(u.erg/u.s/u.cm**2/u.arcsec**2)

#ptsrc_1pix_snr_500s = ((magzero_1pix_count_rate * fiducial_integration_time) /
#                       (readnoise_optimistic + u.Quantity((((magzero_1pix_count_rate
#                                                             + darkn_opt) *
#                                                            fiducial_integration_time)**0.5).decompose().value,
#                                                          u.count))).decompose()
D = dark_rate_optimistic.value
t = fiducial_integration_time.value
snr = 5
R = readnoise_optimistic.value
counts_for_five_sigma = (snr**2 + (snr**4 + 4*(D*t + R**2)*snr**2)**0.5)/2.
countrate_for_five_sigma = counts_for_five_sigma / t * u.count/u.s
#countrate_for_five_sigma = (snr * t * (4*D*t + 4*R*snr + snr**2)**0.5 + 2*snr*R*t + snr**2*t) / (2*t**2) * u.count/u.s
this_is_five = ((countrate_for_five_sigma * fiducial_integration_time) /
                (readnoise_optimistic**2 +
                 u.Quantity((((countrate_for_five_sigma + dark_rate_optimistic) *
                              fiducial_integration_time)).decompose().value,
                            u.count**2))**0.5).decompose()
#countrate_fivesig_to_mag = -2.5*np.log10(countrate_for_five_sigma / throughput / magzero_1pix_count_rate)

def snr_of_countrate(countrate, integration_time=fiducial_integration_time):
    signal = (countrate*integration_time)
    noise = ((dark_rate_optimistic + countrate)*integration_time*u.count + readnoise_optimistic**2)**0.5
    return signal/noise

def countrate_of_snr(snr, integration_time=fiducial_integration_time):
    counts = (snr**2 + (snr**4 + 4*(dark_rate_optimistic*integration_time*u.count + readnoise_optimistic**2).value*snr**2)**0.5)/2. * u.count
    countrate = counts / integration_time
    return countrate

assert countrate_for_five_sigma == countrate_of_snr(5)

for snr in np.arange(1,25):
    np.testing.assert_almost_equal(snr_of_countrate(countrate_of_snr(snr)), snr)
    np.testing.assert_almost_equal(snr_of_countrate(countrate_of_snr(7, integration_time=15*u.s), integration_time=15*u.s), 7)

sb_sensitivity_snr5 = minsignal_for_snr5 = countrate_for_five_sigma / effective_area * e_paa/u.count / pixscale**2
sb_sensitivity_snr10 = minsignal_for_snr10 = countrate_of_snr(10) / effective_area * e_paa/u.count / pixscale**2
sb_sensitivity_snr20 = minsignal_for_snr20 = countrate_of_snr(20) / effective_area * e_paa/u.count / pixscale**2

sb_sensitivity_snr5_MJysr = (minsignal_for_snr5 / paa_bandwidth_Hz)#.to(u.MJy/u.sr, u.spectral_density(wl_paa)))
sb_sensitivity_snr10_MJysr = (minsignal_for_snr10 / paa_bandwidth_Hz)#0.to(u.MJy/u.sr, u.spectral_density(wl_paa)))
sb_sensitivity_snr20_MJysr = (minsignal_for_snr20 / paa_bandwidth_Hz)#0.to(u.MJy/u.sr, u.spectral_density(wl_paa)))

countrate_fivesig_to_mag = -2.5*np.log10(countrate_for_five_sigma / throughput / magzero_1pix_count_rate / ppbeam**0.5)
print('5sig ctrt try 1: ',countrate_fivesig_to_mag)
countrate_fivesig_to_mag = -2.5*np.log10(sb_sensitivity_snr5_MJysr * psf_area * (nu_paa/paa_bandwidth_Hz) / ppbeam**0.5 / mag_zeropoint_paa)
print('5sig ctrt try 2: ',countrate_fivesig_to_mag)

detectionlimit_mag = -2.5*np.log10(minsignal_for_snr5*psf_area / (mag_zeropoint_paa*paa_bandwidth_Hz))


effelsberg_11cm_fwhm = 4.3*u.arcmin
effelsberg_11cm_sb_sens = 20*u.mJy/(effelsberg_11cm_fwhm**2 / (8*np.log(2)) * np.pi)


# wang2010 sensitivity
wang2010_noise = 0.06*u.mJy/u.arcsec**2
hst_paa_width = 187*u.AA # from svo-fps
wang2010_sb_Sens = (wang2010_noise * (hst_paa_width / wl_paa) * nu_paa).to(sb_unit)



# MIRIS FWMH ~ 1865-1885 nm
miris_bandwidth = 20*u.nm
# paper says: "23.7 mJy / (ADU/s)"

# MIRIS rn = 45 e
# MIRIS dark = 0.67 e-/s
darkrate_miris = 0.67*u.count/u.s
rn_miris = 45*u.count
miris_exptime = 20*u.minute
miris_effective_area = np.pi*(80*u.mm / 2)**2
miris_pixscale = 52*u.arcsec
miris_dark_read_noise = (((rn_miris**2 + darkrate_miris*miris_exptime *
                           u.count)**0.5 / miris_exptime * (e_paa/u.count) /
                          miris_effective_area) / miris_pixscale**2)
miris_calculated_sb_sens = miris_dark_read_noise.to(sb_unit)

miris_sb_sens = 0.77*u.mJy/(52*u.arcsec)**2 * (miris_bandwidth / wl_paa) * nu_paa
miris_fwhm = 52*u.arcsec

meergal_fwhm = 0.8*u.arcsec
meergal_sb_sens = 40*u.uJy/(meergal_fwhm**2 / (8*np.log(2)) * np.pi)

thor_fwhm = 10*u.arcsec
thor_sb_sens = 500*u.uJy/(thor_fwhm**2 / (8*np.log(2)) * np.pi)

cornish_fwhm = 1.5*u.arcsec
cornish_las = 15*u.arcsec
cornish_sb_sens = 0.4*u.mJy/(cornish_fwhm**2 / (8*np.log(2)) * np.pi)

vlass_fwhm = 2.5*u.arcsec
vlass_sb_sens = 69*u.uJy/(vlass_fwhm**2 / (8*np.log(2)) * np.pi)

iphas_bandwidth = (100*u.AA / wl_halpha) * nu_halpha
iphas_mag_zeropoint = 2609.81*u.Jy
vphas_fwhm = 1*u.arcsec
vphas_limiting_mag = 20 # 10-sigma
vphas_sens = 10**(vphas_limiting_mag/-2.5) / 10 * iphas_mag_zeropoint * iphas_bandwidth
vphas_psf_area = (vphas_fwhm**2 / (8*np.log(2)) * np.pi * 2)
# divide the point source sensitivity by the PSF area assuming that the S/N of 10 is in the central pixel,
# so the flux is spread over the PSF (this creates a *lower* limiting SB intensity, i.e., better sensitivity)
vphas_sb_sens = (vphas_sens / vphas_psf_area).to(sb_unit)

# 120s exposure times
iphas_fwhm = 1*u.arcsec
iphas_limiting_mag = 20.3 # 5-sigma
iphas_maglim_snr = 5
iphas_sens = 10**(iphas_limiting_mag/-2.5) / iphas_maglim_snr * iphas_mag_zeropoint * iphas_bandwidth
iphas_psf_area = (2*np.pi*iphas_fwhm**2 / (8*np.log(2)))
iphas_sb_sens = (iphas_sens / iphas_psf_area).to(sb_unit)
# <Quantity 0.00452946 MJy / sr>

# from Sabin+ 2014, section 2, paragraph 1
# 50% lower than the above estimate
# This appears to be a typo!!!  The estimate given in the text suggests that dividing this number by 5 should give 1e-17!
# iphas_sb_sens2 = (2.5e-16*u.erg*u.cm**-2/u.s/u.arcsec**2).to(u.MJy/u.sr, u.spectral_density(wl_halpha))
# <Quantity 0.00232904 MJy / sr>
iphas_sb_sens2 = (2.5e-17*u.erg*u.cm**-2/u.s/u.arcsec**2).to(sb_unit)
# <Quantity 0.000232904 MJy / sr>
# Sabin+ 2013 gives a MUCH deeper estimate:
#and the narrow-band component allows the detection of extended optical emission with an Hα surface brightness down to ≃2 × 10−17 erg cm−2 s−1 arcsec−2 (≃3 Rayleighs).
# "3 rayleigh"
iphas_sb_sens3 = (2.5e-17*u.erg*u.cm**-2/u.s/u.arcsec**2).to(sb_unit)
# <Quantity 0.0002329 MJy / sr>
iphas_sb_sens4 = (3*u.Rayleigh * e_halpha / u.ph).to(sb_unit)
# <Quantity 0.00015819 MJy / sr>



# 10800s exposure times;
supercosmos_mag_zeropoint = 2600 * u.Jy # GUESS
#supercosmos_sb_sens = (5*u.Rayleigh / nu_halpha * e_halpha / u.ph).to(u.MJy/u.sr)
supercosmos_sb_sens = (5*u.Rayleigh * e_halpha / u.ph).to(sb_unit)
# <Quantity 0.00026364 MJy / sr>
supercosmos_limiting_mag = 20.5
supercosmos_bandwidth_Hz = (58*u.AA / wl_halpha)*nu_halpha
supercosmos_sens = 10**(supercosmos_limiting_mag/-2.5)/5 * supercosmos_mag_zeropoint * supercosmos_bandwidth_Hz
supercosmos_fwhm = 3*u.arcsec
supercosmos_sb_sens2 = (supercosmos_sens / (supercosmos_fwhm**2 / (8*np.log(2)) * np.pi)).to(sb_unit)
# <Quantity 0.00024188 MJy / sr>


# sanity check
# Say we observe an HII region that has Q_lyc = 1e49 ph/s
Qlyc = 1e49*u.ph/u.s
# it produces paa_to_cont = 0.01 paa photons per lyc photon
paa_to_cont = 0.01019106
# those photons get spread into 4pi steradians, so divide the total
# number of photons by 4 pi distance^2 (assuming it was a sphere/pointsource)
distance = 5 * u.kpc
# we receive effective_area  / 4 pi distance^2 photons per second
rec_flux = (Qlyc * paa_to_cont * effective_area / (4 * np.pi * distance**2)).decompose()
# each pixel gets 1/ppbeam of those, or the peak pixel gets about 1/5 of the photons
peakfrac = (airymod.max() / airymod.sum())
rec_flux_peakpix = peakfrac * rec_flux
# flux to energy requires multiplying by photon energy, so we have erg/s/cm^2
#rec_eflux = rec_flux * e_paa/u.ph
# we assumed this is a point source, so it's getting spread over a PSF
#rec_sb = rec_eflux / psf_area
#print(rec_flux, rec_eflux, rec_sb)
#print((rec_sb / effective_area / paa_bandwidth_Hz).to(u.MJy/u.sr))

def cardelli_law(wavelength, RV=3):
    wavenumber = (wavelength.to(u.um**-1, u.spectral())).value
    a = 0.574 * wavenumber**1.61
    b = -0.527 * wavenumber**1.61
    A_lambda = a + b / RV
    return A_lambda




if __name__ == "__main__":
    #print(supercosmos_sb_sens, supercosmos_sb_sens2, supercosmos_sb_sens/supercosmos_sb_sens2)
    print(f"Optimistic sensitivity:\ndark_rn_opt={dark_rn_opt}\nsb_sensitivity_MJySr={sb_sensitivity_MJySr}")
    print(f"Point source sensitivity in mag: {countrate_fivesig_to_mag}")
    print()

    print(f"5-sigma - requires counts={counts_for_five_sigma}, countrate={countrate_for_five_sigma}")
    print(f"should be 5={this_is_five}")
    print(f"minimum signal for SNR 5 = {minsignal_for_snr5}")
    print(f"minimum signal for SNR 10 = {minsignal_for_snr10}")
    print(f"surf brightness for SNR 5 = {sb_sensitivity_snr5_MJysr}")
    print(f"surf brightness for SNR 10 = {sb_sensitivity_snr10_MJysr}")

    print()
    print(f"minimum signal for SNR 5 / (5*calculated_sb_sens) = {minsignal_for_snr5/(5*calculated_sb_sens)}")
    print(f"minimum signal for SNR 10 / (10*calculated_sb_sens) = {minsignal_for_snr10/(10*calculated_sb_sens)}")
    print(f"minimum signal for SNR 20 / (20*calculated_sb_sens) = {minsignal_for_snr20/(20*calculated_sb_sens)}")

    print()
    print(f"sb_sens={sb_sensitivity}")
    print(f"calculated_sb_sens={calculated_sb_sens}")
    print()

    print(f"Maximum unsaturated magnitude ({fiducial_integration_time}): {max_unsaturated_mag}")
    print(f"Maximum unsaturated flux ({fiducial_integration_time}): {max_unsaturated_flux}")
    print(f"Maximum unsaturated sb ({fiducial_integration_time}): {max_unsaturated_sb}")
    print(f"Maximum unsaturated surface brightness ({fiducial_integration_time}): {max_unsaturated_sb}")
