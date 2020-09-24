import numpy as np
from astropy import units as u
from astropy import constants
import sensitivity

def lacc(mdot, rstar=u.R_sun, mstar=u.M_sun):
    return (mdot / 1.25 * constants.G * mstar / rstar).to(u.L_sun)

def log_pab(lacc):
    return 1/1.06 * np.log10(lacc/u.L_sun) - 2.76/1.06

def log_paa(lacc):
    return log_pab(lacc) + np.log10(2.86 * 0.336 / 0.347)

def L_paa(lacc):
    return 10**log_paa(lacc) * u.L_sun

def S_paa(lacc, distance=8*u.kpc):
    # I think this is wrong; I probably got a unit wrong above
    return (L_paa(lacc) / (4*np.pi*distance**2)).to(u.erg/u.s/u.cm**2)

if __name__ == "__main__":

    # Ekstrom 2012 evolutionary models; we'll use the age=10^6.5 for zero-age radii, lums

    from astropy import visualization
    visualization.quantity_support()
    import pylab as pl
    pl.ion()
    from astroquery.vizier import Vizier
    tbl = Vizier(row_limit=1e7, columns=['**']).get_catalogs('J/A+A/537/A146/iso')[0]

    zeroage = tbl['logAge'] == 6.5
    masses = tbl['Mini'][zeroage].quantity
    radius = tbl['Rpol'][zeroage].quantity
    #pl.plot(masses, radius)

    pl.clf()
    mdot = np.logspace(-10, -5)*u.M_sun/u.yr
    #for mass, rad in list(zip(masses, radius))[::230]:
    #for mass in [1, 2, 4, 6, 8, 10, 15, 20]*u.M_sun:
    #    closest = np.argmin(np.abs(masses-mass))
    #    rad = radius[closest]
    #    acclum = lacc(mdot, rstar=rad, mstar=mass)
    #    paalum = S_paa(acclum)
    #    pl.loglog(mdot, paalum, label=f'R={rad.to(u.R_sun):0.1f} M={mass.to(u.M_sun):0.1f}')

    distance = np.logspace(-0.1, 2.1)*u.kpc
    distance = np.geomspace(0.8, 50)*u.kpc
    a_v = distance * 2 / u.kpc
    attenuation = 10**(-a_v * 0.15 / 2.5)
    for mdot in 10.**np.arange(-8, -4, 1)*u.M_sun/u.yr:
        acclum = lacc(mdot, rstar=1*u.R_sun, mstar=u.M_sun)
        paalum = S_paa(acclum, distance=distance)

        pl.loglog(distance, paalum * attenuation, label=f'$\\log \\dot{{M}}={int(np.log10(mdot.value))}$')
    pl.loglog(distance, paalum, linestyle=":", label=f'$\\log \\dot{{M}}={int(np.log10(mdot.value))}~(A_V=0)$')

    # point source sensitivity
    pl.hlines(np.array([5,20])*sensitivity.ps_sensitivity,
              distance.min(), distance.max(), linestyle='--', color='k')
    yl = pl.ylim()
    pl.fill_betweenx(yl, 3, 15, color='k', alpha=0.05)
    pl.ylim(1e-15, 1e-10)
    pl.xlim(0.9, 50)

    pl.xlabel("Distance [kpc]")
    pl.ylabel("Source Flux [erg s$^{-1}$ cm$^{-2}$]")

    pl.legend(loc='best', fontsize=14)
    pl.savefig("figures/accretion_rate_sensitivity.pdf", bbox_inches='tight')
