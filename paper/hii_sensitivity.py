import os
import numpy as np
import pylab as pl
from astropy import units as u
from pyspeckit.spectrum.models import hydrogen
from astropy import constants

from sensitivity import ps_sensitivity, dark_rn_opt, e_paa, nu_paa, wl_paa, psf_area, sb_sensitivity, sb_sensitivity_snr5, sb_sensitivity_snr10, sb_unit, cardelli_law

from astropy.visualization import quantity_support
quantity_support()

pl.rcParams['font.size'] = 16

# from table in Draine
alpha_b_5e3 = 4.53e-13*u.cm**3*u.s**-1
alpha_b_1e4 = 2.59e-13*u.cm**3*u.s**-1
alpha_b_2e4 = 1.43e-13*u.cm**3*u.s**-1
alpha_b_beta_1e4 = 3.03e-14*u.cm**3*u.s**-1

wl_hbeta = hydrogen.wavelength['balmerb']*u.um
nu_hbeta = wl_hbeta.to(u.Hz, u.spectral())

wl_halpha = hydrogen.wavelength['balmera']*u.um
nu_halpha = wl_halpha.to(u.Hz, u.spectral())
e_halpha = wl_halpha.to(u.erg, u.spectral())


def alpha_eff(Te=1e4*u.K, line='beta'):
    """ H-alpha recombination coefficient.  eqn 14.8, 14.9 in draine 2001"""
    T4 = (Te/(1e4*u.K)).decompose().value
    if line == 'alpha':
        return 1.17e-13 * T4**(-0.942-0.031*np.log(T4)) * u.cm**3*u.s**-1
    elif line == 'beta':
        return 3.03e-14 * T4**(-0.874-0.058*np.log(T4)) * u.cm**3*u.s**-1

# ratio of paa to hbeta
paa_to_hb_1e4 = 0.336
# ratio of H-alpha to H-beta
ha_to_hb_1e4 = 2.86
hg_to_hb_1e4 = 0.469
pab_to_hgamma_1e4 = 0.347

def alpha_paa(Te=1e4*u.K):
    return alpha_eff(Te, line='beta') * paa_to_hb_1e4 * (wl_hbeta / wl_paa)

ha_to_paa_1e4 = ha_to_hb_1e4 / paa_to_hb_1e4
ha_to_paa_1e4_phots = (e_halpha/e_paa)**-1 * ha_to_paa_1e4


def EMfunc(Qlyc=1e45*u.s**-1, R=0.1*u.pc, alpha_b=2e-13*u.cm**3*u.s**-1):
    return (R * (((3 * Qlyc)/(4 * np.pi * R**3 * alpha_b))**0.5)**2).to(u.cm**-6*u.pc)

def dens(Qlyc=1e45*u.s**-1, R=0.1*u.pc, alpha_b=2e-13*u.cm**3*u.s**-1):
    return (((3 * Qlyc)/(4 * np.pi * R**3 * alpha_b))**0.5).to(u.cm**-3)

def snu_halpha(Te=10000*u.K, EM=EMfunc(alpha_b=alpha_b_1e4), angular_area=4*np.pi*u.sr):
    jhb_4p = 1.24e-25 * u.erg * u.cm**3 / u.s
    alpha_rel = alpha_eff(Te=Te, line='beta') / alpha_b_beta_1e4
    assert alpha_eff(Te=1e4*u.K, line='beta') == alpha_b_beta_1e4
    flux = EM * jhb_4p * ha_to_hb_1e4 * alpha_rel
    return flux / angular_area


def snu_hbeta(Te=10000*u.K, EM=EMfunc(alpha_b=alpha_b_1e4), angular_area=4*np.pi*u.sr):
    jhb_4p = 1.24e-25 * u.erg * u.cm**3 / u.s
    flux = EM * jhb_4p
    return flux / angular_area

def snu_paa(Te=10000*u.K, EM=EMfunc(alpha_b=alpha_b_1e4), angular_area=4*np.pi*u.sr):
    # temperature dependence factor: jhb ~ alpha, so this accounts for the T-dependence of j
    # (which is not explicitly given in Draine)
    alpha_rel = alpha_eff(Te=Te, line='beta') / alpha_b_beta_1e4
    jhb_4p = 1.24e-25 * u.erg * u.cm**3 / u.s * paa_to_hb_1e4 * alpha_rel
    flux = EM * jhb_4p
    return flux / angular_area

def snu_paa_try2(Te=1e4*u.K, EM=EMfunc(alpha_b=alpha_b_1e4), angular_area=4*np.pi*u.sr):
    """
    this is the version where I try to derive the intensity jPa from alpha_pa
    """
    jpa_4p = alpha_paa(Te=Te) * e_paa * 4*np.pi
    assert jpa_4p.unit.is_equivalent(u.erg * u.cm**3 / u.s)
    flux = EM * jpa_4p
    return flux / angular_area

def em_of_snu_paa(snu_per_sr, Te=1e4*u.K, angular_area=4*np.pi*u.sr):
    alpha_rel = alpha_eff(Te=Te, line='beta') / alpha_b_beta_1e4
    jpa_4p = 1.24e-25 * u.erg * u.cm**3 / u.s * paa_to_hb_1e4 * alpha_rel
    EM = (snu_per_sr).to(sb_unit) * angular_area / jpa_4p
    return EM.to(u.cm**-6*u.pc)

def em_of_snu_halpha(snu_per_sr, Te=1e4*u.K, angular_area=4*np.pi*u.sr):
    alpha_rel = alpha_eff(Te=Te, line='alpha') / alpha_b_beta_1e4
    jha_4p = 1.24e-25 * u.erg * u.cm**3 / u.s * ha_to_hb_1e4 * alpha_rel
    EM = (snu_per_sr).to(sb_unit) * angular_area / jha_4p
    return EM.to(u.cm**-6*u.pc)


def gff(nu, Te=1e4*u.K, Zi=1):
    """
    Draine eqn 10.7
    """
    valid_check =  (Zi * (nu/u.GHz) / (Te/(1e4*u.K))**1.5)
    if np.any(valid_check < 0.14):
        raise ValueError("Invalid approximation")
    elif np.any(valid_check > 250):
        # eqn 10.9
        return np.log(np.exp(5.960-3**0.5/np.pi*np.log(Zi*(nu/u.GHz)*(Te/(1e4*u.K))**-1.5)) + np.exp(1))
    return 6.155 * (Zi * (nu/u.GHz))**-0.118 * (Te/(1e4*u.K))**0.177


def freefree_draine_coeff(nu, Te=1e4*u.K, Zi=1):
    """
    Draine eqn 10.1, 10.2
    """
    const = 8/3 * (2*np.pi/3)**0.5 * constants.e.esu**6/constants.m_e**2/constants.c**3 * (constants.m_e/(constants.k_B*Te))**0.5
    unit = u.erg/u.s/u.Hz*u.cm**3
    assert np.abs(const.to(unit) - 5.444e-41*unit) < 0.001e-41*unit
    gaunt = gff(nu=nu, Te=Te)
    boltz = np.exp(-constants.h*nu / (constants.k_B*Te))
    temfac = (Te/(1e4*u.K))**-0.5
    # this is multiplied by density squared
    return const.to(unit) * boltz * temfac * u.sr**-1 * Zi**2 * gaunt

assert np.abs(freefree_draine_coeff(1*u.GHz) - 3.35e-40*u.erg*u.cm**3/u.s/u.Hz/u.sr).value < 0.01e-40

def freefree_draine(nu, EM, Te=1e4*u.K):
    coef = freefree_draine_coeff(nu=nu, Te=Te)
    assert coef.unit.is_equivalent(u.cm**3*u.erg/u.Hz/u.s/u.sr)
    return (coef*EM).to(u.erg/u.s/u.cm**2/u.Hz/u.sr)


def em_of_snu_freefree(nu, snu, Te=1e4*u.K):
    coef = freefree_draine_coeff(nu=nu, Te=Te)
    assert coef.unit.is_equivalent(u.cm**3*u.erg/u.Hz/u.s/u.sr)
    EM = (snu/coef).to(u.cm**-6*u.pc)
    return EM



if __name__ == "__main__":
    paa_to_cont = alpha_paa() / alpha_b_1e4

    # A_halpha/A_V * A_V - A_paa/A_V * A_V = m_halpha - m_paa = -2.5 np.log10 (ha_to_paa_1e4_phots)
    # A_V = -2.5*np.log10(ha(cl_ha - cl_paa)
    mag_paa_brighter = 2.5*np.log10(ha_to_paa_1e4_phots) / (cardelli_law(wl_halpha) - cardelli_law(wl_paa))
    print(f"Magnitude at which PaA is bright is {mag_paa_brighter}")

    np.testing.assert_almost_equal(snu_paa(Te=1e4*u.K, EM=1e6*u.cm**-6*u.pc).to(u.erg/u.s/u.cm**2/u.sr).value,
                                   0.01023061)

    # validation plot: check that the cataloged values for alpha_b match those inferred from the fit eqn above
    tems = np.linspace(3000,20000)
    fig1 = pl.figure(1)
    fig1.clf()
    pl.semilogy(tems, alpha_paa(Te=tems))
    pl.plot([5000,1e4,2e4], u.Quantity([alpha_b_5e3, alpha_b_1e4, alpha_b_2e4]) * paa_to_cont, 's')

    qlycs = np.logspace(40,52)*u.s**-1
    radii = np.logspace(0.1,2,10)*u.pc

    fig = pl.figure(2, figsize=(12,12))
    fig.clf()
    for radius in radii:
        n_atoms = dens(Qlyc=qlycs, R=radius) * 4/3*np.pi*radius**3
        pl.loglog(qlycs, (dens(Qlyc=qlycs, R=radius)*alpha_paa()*n_atoms).to(u.s**-1), label=radius)
    pl.loglog([1e41,1e51],[1e41,1e51])
    pl.ylabel("N(photons)/s")
    pl.xlabel("$Q_{lyc}$")


    fig3 = pl.figure(3, figsize=(10,8))
    fig3.clf()
    qlycs = [1e45, 1e46, 1e47, 1e48, 1e49]*u.s**-1
    paas = paa_to_cont * qlycs * e_paa
    distances = np.logspace(0, 2)*u.kpc

    a_v = distances * 2 / u.kpc
    attenuation = 10**(-a_v * 0.15 / 2.5)

    for paa,ql in zip(paas, qlycs):
        pl.loglog(distances, (paa/distances**2).to(u.erg/u.s/u.cm**2) / (4*np.pi) * attenuation, label=f"$10^{{{int(np.log10(ql.value))}}}$ ph/s")
    pl.loglog(distances, (paa/distances**2).to(u.erg/u.s/u.cm**2) / (4*np.pi), label=f"$10^{{{int(np.log10(ql.value))}}}$ ph/s $A_V=0$")
    pl.hlines(u.Quantity([ps_sensitivity*5, ps_sensitivity*10]), 1, 100, color='k', linestyle=':')

    yl = pl.ylim()
    pl.fill_betweenx(yl, 3, 15, color='k', alpha=0.05)
    pl.ylim(1e-15, 1e-10)
    pl.xlim(1, 50)

    pl.ylabel("S$_{Pa\\alpha}$ [erg s$^{-1}$ cm$^{-2}$]")
    pl.xlabel("Distance (kpc)")
    pl.legend(loc='best')
    pl.savefig("figures/HII_region_sensitivity.pdf", bbox_inches='tight')



    fig4 = pl.figure(4, figsize=(10,8))
    fig4.clf()
    ax4 = fig4.gca()


    Qlyc = np.logspace(44,49,100)*u.s**-1
    for radius in (0.05, 0.15, 0.5, 1.5)*u.pc:
        EMs = EMfunc(R=radius, Qlyc=Qlyc)
        snus = snu_paa(Te=1e4*u.K, EM=EMs).to(sb_unit)
        ax4.loglog(Qlyc, snus, label=f'R={radius:0.2f}')
    pl.legend(loc='best')
    #pl.hlines(u.Quantity([dark_rn_opt, dark_rn_pess])*5, Qlyc.min(), Qlyc.max(), linestyle='--', color='k')
    fivesig = sb_sensitivity_snr5
    tensig = sb_sensitivity_snr10
    bgcolor=pl.gcf().get_facecolor()
    color = (1-bgcolor[0], 1-bgcolor[1], 1-bgcolor[2], bgcolor[3])
    lines = ax4.hlines(u.Quantity([fivesig, tensig]), Qlyc.min(), Qlyc.max(), linestyle='--', color=color)
    ax4.set_xlim(1e44,1e49)
    pl.ylim(dark_rn_opt.value, pl.gca().get_ylim()[1])
    pl.xlabel("Q$_{LyC}$ [ph s$^{-1}$]")
    pl.ylabel("Pa$\\alpha$ Surface Brightness\n[erg s$^{-1}$ cm$^{-2}$ as$^{-2}$]")
    fig4.savefig(os.path.expanduser("figures/extended_HII_region_detectability.pdf"), bbox_inches='tight')

    #not a fair comparison
    # fig5 = pl.figure(5, figsize=(10,8))
    # fig5.clf()
    # ax5 = fig5.gca()
    # EMs = np.logspace(0,7)*u.cm**-6*u.pc
    # ax5.loglog(EMs, (freefree_draine(nu=5*u.GHz, Te=1e4*u.K, EM=EMs) * psf_area).to(u.uJy), label='5 GHz free-free')
    # ax5.loglog(EMs, snu_paa(Te=1e4*u.K, EM=EMs)*psf_area, label='Pa$\\alpha$')
    # ax5.hlines(np.array([1,5,20])*ps_sensitivity/nu_paa, EMs.min(), EMs.max(), color='k', linestyle='--')
    # ax5.set_xlabel(f"Emission Measure [{EMs.unit}]")
    # yl = ax5.get_ylabel()
    # ax5.set_ylabel(f"Flux Density [{yl}]")
    # pl.legend(loc='best')
