from astroquery.vizier import Vizier
import numpy as np
from astropy import units as u
import imf
from astropy.utils.console import ProgressBar

Vizier.ROW_LIMIT = 1e7 # effectively infinite

# this query should cache
tbl = Vizier(row_limit=1e7, columns=['**']).get_catalogs('J/A+A/537/A146/iso')[0]

wolfrayet = tbl[(-7 < tbl['logdM_dtr']) & (tbl['logdM_dtr'] < 0) & (tbl['logTe'] > 4.1)]

logages = np.array(sorted(list(set(tbl['logAge']))))
maxmass_of_age = [tbl['Mass'][tbl['logAge']==age].max() for age in logages]

ages = (10**logages)*u.yr
agelims = dict(zip(ages, maxmass_of_age))

def maxmass(age, mmax=120):
    return np.interp(u.Quantity(age, u.yr), u.Quantity(ages, u.yr),
                     maxmass_of_age, left=mmax)

sfr = 1*u.Msun/u.yr

age_ostar = 3*u.Myr



agebins = np.linspace(0, 3e7, 1000)*u.yr
dage = np.diff(agebins)[0]

nsamps = 11
ostars_of_age = {}
bstars_of_age = {}
abstars_of_age = {}
for age in ProgressBar(agebins):
    clusters = [imf.make_cluster((sfr * dage).to(u.M_sun).value, silent=True)
                for ii in range(nsamps)]
    mmax = maxmass(age)
    total_ostars = [((cl > 20) & (cl < mmax)).sum() for cl in clusters]
    total_bstars = [((cl > 8) & (cl < 20) & (cl < mmax)).sum() for cl in clusters]
    total_abstars = [((cl > 2) & (cl < 8)).sum() for cl in clusters]
    med_tot_o = np.median(total_ostars)
    med_tot_b = np.median(total_bstars)
    med_tot_ab = np.median(total_abstars)
    #print(f"Total O-stars={med_tot} for age {age}")
    ostars_of_age[age] = med_tot_o
    bstars_of_age[age] = med_tot_b
    abstars_of_age[age] = med_tot_ab

print(f"Total ostars now: {np.sum(list(ostars_of_age.values()))}")
print(f"Total bstars now: {np.sum(list(bstars_of_age.values()))}")
print(f"Total abstars now: {np.sum(list(abstars_of_age.values()))}")



yso_age = 0.5*u.Myr
yso_mmin = 2
clusters = [imf.make_cluster((sfr * yso_age).to(u.M_sun).value, silent=True)
            for ii in range(nsamps)]
n_detectable_ysos = np.median([(cl > yso_mmin).sum() for cl in clusters])

print(f"Detectable YSOs >{yso_mmin} Msun: {n_detectable_ysos}")
