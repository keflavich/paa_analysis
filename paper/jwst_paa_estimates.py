import numpy as np
from astropy import units as u
from astropy import constants
from sensitivity import cardelli_law
from pyspeckit.spectrum.models import hydrogen
from hii_sensitivity import ha_to_hb_1e4, paa_to_hb_1e4, bra_to_hgamma_1e4, pab_to_hgamma_1e4, hg_to_hb_1e4, wl_paa
from astroquery.svo_fps import SvoFps

wl_bra = hydrogen.wavelength['bracketta']*u.um
twomass = SvoFps.get_filter_list('2MASS')
wl_Ks = twomass[twomass['filterID'].astype(str)=='2MASS/2MASS.Ks']['WavelengthCen'][0] * u.AA

def lacc(mdot, rstar=u.R_sun, mstar=u.M_sun):
    # Alcala+ 2017 eqn 1
    # 1.25 comes from assuming r_inner disk = 5 r_star
    return (mdot / 1.25 * constants.G * mstar / rstar).to(u.L_sun)

def log_pab(lacc):
    # from table B1 of Alcala+2017:
    # log L_acc = 1.06 log L_pab + 2.76
    # so
    # log L_pab = (log L_acc - 2.76) / 1.06
    return 1/1.06 * np.log10(lacc/u.L_sun) - 2.76/1.06

def log_paa(lacc):
    # pab * (paa / hb) * (hb / hg) *  (hg / pab)
    return log_pab(lacc) + np.log10(paa_to_hb_1e4 / hg_to_hb_1e4 / pab_to_hgamma_1e4)

def log_bra(lacc):
    # brackett alpha prediction from PaB (could use BrG too...)
    return log_pab(lacc) + np.log10(bra_to_hgamma_1e4 / pab_to_hgamma_1e4)

def L_paa(lacc):
    return 10**log_paa(lacc) * u.L_sun

def L_bra(lacc):
    return 10**log_bra(lacc) * u.L_sun

def S_paa(lacc, distance=8*u.kpc, A_K=0):
    A_paa = (cardelli_law(wl_paa) / cardelli_law(wl_Ks)) * A_K
    attenuation = 10**(-A_paa / 2.5)
    return (L_paa(lacc) / (4*np.pi*distance**2)).to(u.erg/u.s/u.cm**2) * attenuation

def S_bra(lacc, distance=8*u.kpc, A_K=0):
    A_bra = (cardelli_law(wl_bra) / cardelli_law(wl_Ks)) * A_K
    attenuation = 10**(-A_bra / 2.5)
    return (L_bra(lacc) / (4*np.pi*distance**2)).to(u.erg/u.s/u.cm**2) * attenuation


if __name__ == "__main__":

    # Ekstrom 2012 evolutionary models; we'll use the age=10^6.5 for zero-age radii, lums

    from astropy import visualization
    visualization.quantity_support()
    import pylab as pl
    pl.ion()
    #from astroquery.vizier import Vizier
    #tbl = Vizier(row_limit=1e7, columns=['**']).get_catalogs('J/A+A/537/A146/iso')[0]

    jwst_paa_tr = SvoFps.get_transmission_data('JWST/NIRCam.F187N')
    jwst_paa_effectivewidth = (np.diff(jwst_paa_tr['Wavelength'].quantity) * jwst_paa_tr['Transmission'].quantity[1:]).sum() / jwst_paa_tr['Transmission'].quantity[1:].max()
    jwst_paa_central_wavelength = (jwst_paa_tr['Wavelength'].quantity * jwst_paa_tr['Transmission'].quantity).sum() / jwst_paa_tr['Transmission'].quantity.sum()
    jwst_paa_effectivewidth_hz = (jwst_paa_effectivewidth / jwst_paa_central_wavelength) * jwst_paa_central_wavelength.to(u.Hz, u.spectral())

    jwst_bra_tr = SvoFps.get_transmission_data('JWST/NIRCam.F405N')
    jwst_bra_effectivewidth = (np.diff(jwst_bra_tr['Wavelength'].quantity) * jwst_bra_tr['Transmission'].quantity[1:]).sum() / jwst_bra_tr['Transmission'].quantity[1:].max()
    jwst_bra_central_wavelength = (jwst_bra_tr['Wavelength'].quantity * jwst_bra_tr['Transmission'].quantity).sum() / jwst_bra_tr['Transmission'].quantity.sum()
    jwst_bra_effectivewidth_hz = (jwst_bra_effectivewidth / jwst_bra_central_wavelength) * jwst_bra_central_wavelength.to(u.Hz, u.spectral())


    pl.figure(1)
    pl.clf()
    mdot = np.logspace(-10, -5)*u.M_sun/u.yr
    lacc_vals = lacc(mdot)
    spaa = S_paa(lacc_vals, distance=8.2*u.kpc)
    sbra = S_bra(lacc_vals, distance=8.2*u.kpc)

    # values computed for 24 exposures RAPID 2 group
    limit_10sigma_paa = 2.7e-17 * u.erg/u.s/u.cm**2
    limit_10sigma_bra = 4.25e-17 * u.erg/u.s/u.cm**2

    A_K = 2.5
    A_paa = (cardelli_law(wl_paa) / cardelli_law(wl_Ks)) * A_K
    A_bra = (cardelli_law(wl_bra) / cardelli_law(wl_Ks)) * A_K
    print(f"PaA extinction:{A_paa},  BrA extinction:{A_bra} (For A_K = 2.5 mag)")

    # mJy if we want to include it in the filters
    spaa_mJy = (spaa / jwst_paa_effectivewidth_hz).to(u.mJy)
    sbra_mJy = (sbra / jwst_paa_effectivewidth_hz).to(u.mJy)

    pl.loglog(mdot, spaa * 10**(-A_paa/2.5), label='Pa$\\alpha$')
    pl.loglog(mdot, sbra * 10**(-A_bra/2.5), label='Br$\\alpha$')

    pl.xlabel(f'$\\dot{{M}} [M_\odot]$')
    pl.ylabel("Source Flux [erg s$^{-1}$ cm$^{-2}$]")
    pl.legend(loc='best')

    #mag_paa_brighter = 2.5*np.log10(ha_to_paa_1e4_phots) / (cardelli_law(wl_halpha) - cardelli_law(wl_paa))
    #print(f"Magnitude at which PaA is brighter than H-alpha is {mag_paa_brighter}")


    print(cardelli_law(wl_bra) - cardelli_law(wl_paa))


    pl.figure(2)
    pl.clf()
    ax = pl.gca()

    A_Ks = np.linspace(1, 40, 100)
    for mdot in (1e-8, 1e-6, 1e-4)*u.M_sun/u.yr:
        line, = pl.loglog(A_Ks, S_paa(lacc(mdot), distance=8.2*u.kpc, A_K=A_Ks), color='orange')#label='Pa$\\alpha$')
        line, = pl.loglog(A_Ks, S_bra(lacc(mdot), distance=8.2*u.kpc, A_K=A_Ks), color='blue')#label='Br$\\alpha$')
    ax.annotate("$\dot{M}=10^{-8} \mathrm{M}_\odot$", (2, 1e-16, ))
    ax.annotate("$\dot{M}=10^{-6} \mathrm{M}_\odot$", (2, 1e-14, ))
    ax.annotate("$\dot{M}=10^{-4} \mathrm{M}_\odot$", (2, 5e-13, ))

    pl.axhline(limit_10sigma_paa, linestyle='--', color='orange', label='Pa$\\alpha$')
    pl.axhline(limit_10sigma_bra, linestyle=':', color='blue', label='Br$\\alpha$')

    ax.set_ylim(1e-17, 2e-12)
    ax.set_xlim(2,30)

    pl.xlabel(f'$A_K$ [mag]')
    pl.ylabel("Source Flux [erg s$^{-1}$ cm$^{-2}$]")
    pl.legend(loc='best')
