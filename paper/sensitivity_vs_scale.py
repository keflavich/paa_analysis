import os
import numpy as np
import pylab as pl
from astropy import units as u
from pyspeckit.spectrum.models import hydrogen
from astropy import constants

from sensitivity import (ps_sensitivity, dark_rn_opt, e_paa, nu_paa, wl_paa,
                         psf_area, sb_sensitivity, fwhm,
                         sb_sensitivity_snr5,
                         sb_sensitivity_snr10,
                         miris_fwhm, miris_sb_sens,
                         meergal_fwhm, meergal_sb_sens,
                         thor_fwhm, thor_sb_sens,
                         cornish_fwhm, cornish_sb_sens, cornish_las,
                         effelsberg_11cm_fwhm, effelsberg_11cm_sb_sens,
                         vlass_fwhm, vlass_sb_sens,
                         iphas_fwhm, iphas_sb_sens4,
                         vphas_fwhm, vphas_sb_sens,
                         supercosmos_fwhm, supercosmos_sb_sens,
                        )

from hii_sensitivity import em_of_snu_paa, em_of_snu_freefree, em_of_snu_halpha

from astropy.visualization import quantity_support
quantity_support()

pl.rcParams['font.size'] = 16

pl.figure(1).clf()


xaxis = np.linspace(0.5, 5000, 5000)*u.arcsec

pashion_sens = em_of_snu_paa(sb_sensitivity_snr5)
resoln_ok = xaxis > fwhm

sens = (pashion_sens * fwhm / xaxis)[resoln_ok]
ax = pl.gca()
ax.cla()
ax.loglog()
ax.fill_between(xaxis[resoln_ok], sens, np.ones(resoln_ok.sum())*pashion_sens*10000, label='PASHION', facecolor='none', edgecolor='k')


#MEERGAL is an SKA program, not a MEERKAT program.
meergal_resoln_ok = (xaxis > meergal_fwhm) & (xaxis < 500*u.arcsec)
meergal_sens = (em_of_snu_freefree(14*u.GHz, meergal_sb_sens) * (meergal_fwhm / xaxis)**0.1)[meergal_resoln_ok]
ax.fill_between(xaxis[meergal_resoln_ok], meergal_sens*5, np.ones(meergal_resoln_ok.sum())*meergal_sens*1000, alpha=0.5, label='MEERGAL', facecolor='none', edgecolor='green', linestyle='--')

thor_resoln_ok = (xaxis > thor_fwhm) & (xaxis < 970*u.arcsec)
thor_sens = (em_of_snu_freefree(1.4*u.GHz, thor_sb_sens) * (thor_fwhm / xaxis)**0.1)[thor_resoln_ok]
ax.fill_between(xaxis[thor_resoln_ok], thor_sens*5, np.ones(thor_resoln_ok.sum())*thor_sens*100000, alpha=0.5, label='THOR', facecolor='none', edgecolor='blue', linestyle='--')

cornish_resoln_ok = (xaxis > cornish_fwhm) & (xaxis < cornish_las)
cornish_sens = (em_of_snu_freefree(5*u.GHz, cornish_sb_sens) * (cornish_fwhm / xaxis)**0.1)[cornish_resoln_ok]
ax.fill_between(xaxis[cornish_resoln_ok], cornish_sens*5, np.ones(cornish_resoln_ok.sum())*cornish_sens*100000, alpha=0.5, label='CORNISH', facecolor='none', edgecolor='orange', linestyle='--')


vlass_las = 58*u.arcsec
vlass_resoln_ok = (xaxis > vlass_fwhm) & (xaxis < vlass_las)
vlass_sens = (em_of_snu_freefree(3*u.GHz, vlass_sb_sens) * (vlass_fwhm / xaxis)**0.1)[vlass_resoln_ok]
ax.fill_between(xaxis[vlass_resoln_ok], vlass_sens*5, np.ones(vlass_resoln_ok.sum())*vlass_sens*100000, alpha=0.5, label='VLASS', facecolor='none', edgecolor='maroon', linestyle='--')

effelsberg_11cm_resoln_ok = (xaxis > effelsberg_11cm_fwhm)
effelsberg_11cm_sens = (em_of_snu_freefree((11*u.cm).to(u.GHz, u.spectral()), effelsberg_11cm_sb_sens) * (effelsberg_11cm_fwhm / xaxis))[effelsberg_11cm_resoln_ok]
ax.fill_between(xaxis[effelsberg_11cm_resoln_ok], effelsberg_11cm_sens*5, np.ones(effelsberg_11cm_resoln_ok.sum())*effelsberg_11cm_sens*100000,
                alpha=0.5, label='Bonn 11cm', facecolor='none', edgecolor='navy', linestyle=':')



ax.set_ylim(0.2, 1e6)
ax.set_xlim(1, 4000)

leg = pl.legend(loc='lower left', fontsize=12)

ax.set_title("PASHION vs Radio Surveys")
ax.set_xlabel("Angular Size (arcseconds)")
ax.set_ylabel("Sensitivity to EM [cm$^{-6}$ pc]")

pl.savefig("figures/radio_survey_comparison.png", bbox_inches='tight')



pl.figure(2).clf()

a_v = 10
attenuation_paa = 10**(-a_v*0.15/2.5)
attenuation_halpha = 10**(-a_v/2.5)

pashion_sens = em_of_snu_paa(sb_sensitivity_snr5) / attenuation_paa
resoln_ok = xaxis > fwhm

sens = (pashion_sens * fwhm / xaxis)[resoln_ok]
ax = pl.gca()
ax.cla()
ax.loglog()
ax.set_title(f"PASHION vs Optical Surveys at $A_V={a_v}$")
ax.fill_between(xaxis[resoln_ok], sens, np.ones(resoln_ok.sum())*pashion_sens*1000, label='PASHION', facecolor='none', edgecolor='k')


miris_resoln_ok = xaxis > miris_fwhm
miris_sens = (em_of_snu_paa(miris_sb_sens) * miris_fwhm / xaxis)[miris_resoln_ok] / attenuation_paa
ax.fill_between(xaxis[miris_resoln_ok], miris_sens*5, np.ones(miris_resoln_ok.sum())*miris_sens*100000, alpha=0.5, label='MIRIS', facecolor='none', edgecolor='red')

iphas_resoln_ok = (xaxis > iphas_fwhm)
iphas_sens = (em_of_snu_halpha(iphas_sb_sens4) * iphas_fwhm / xaxis)[iphas_resoln_ok] / attenuation_halpha
ax.fill_between(xaxis[iphas_resoln_ok], iphas_sens*5, np.ones(iphas_resoln_ok.sum())*iphas_sens*100000, alpha=0.5, label='IPHAS', facecolor='none', edgecolor='darkgreen', linestyle='--')

#vphas_resoln_ok = (xaxis > vphas_fwhm)
#vphas_sens = (em_of_snu_halpha(vphas_sb_sens) * vphas_fwhm / xaxis)[vphas_resoln_ok] / attenuation_halpha
#ax.fill_between(xaxis[vphas_resoln_ok], vphas_sens, np.ones(vphas_resoln_ok.sum())*vphas_sens*100000, alpha=0.5, label='VPHAS', facecolor='none', edgecolor='cornflowerblue', linestyle='--')

supercosmos_resoln_ok = (xaxis > supercosmos_fwhm)
supercosmos_sens = (em_of_snu_halpha(supercosmos_sb_sens) * supercosmos_fwhm / xaxis)[supercosmos_resoln_ok] / attenuation_halpha
ax.fill_between(xaxis[supercosmos_resoln_ok], supercosmos_sens*5,
                np.ones(supercosmos_resoln_ok.sum())*supercosmos_sens*1e6,
                alpha=0.5, label='SuperCOSMOS', facecolor='none',
                edgecolor='purple', linestyle='--')

ax.set_ylim(10., 1e5)
ax.set_xlim(1, 200)

pl.legend(loc='lower left')

ax.set_xlabel("Angular Size (arcseconds)")
ax.set_ylabel("Sensitivity to EM [cm$^{-6}$ pc]")

pl.savefig("figures/optical_survey_comparison.png", bbox_inches='tight')
