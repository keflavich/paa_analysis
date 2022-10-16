import numpy as np
import glob
from astropy import table
from astropy.io import fits
from astropy import units as u
from astropy import constants
from astropy.utils.console import ProgressBar
from spectral_cube import lower_dimensional_structures
import requests
import tarfile
import os

url = "http://specmodels.iag.usp.br/fits_search/compress/s_coelho14_sed.tgz"
fn = os.path.basename(url)
if not os.path.exists(fn):
    resp = requests.get(url)
    with open(fn, 'wb') as fh:
        fh.write(resp.content)

dir = os.path.splitext(fn)[0]
if not os.path.exists(dir):
    with tarfile.open(fn) as tf:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf)

from astroquery.svo_fps import SvoFps
jcurve = SvoFps.get_transmission_data('2MASS/2MASS.J')
hcurve = SvoFps.get_transmission_data('2MASS/2MASS.H')
kcurve = SvoFps.get_transmission_data('2MASS/2MASS.Ks')
s36curve = SvoFps.get_transmission_data('Spitzer/IRAC.I1')
s45curve = SvoFps.get_transmission_data('Spitzer/IRAC.I2')
s58curve = SvoFps.get_transmission_data('Spitzer/IRAC.I3')
s80curve = SvoFps.get_transmission_data('Spitzer/IRAC.I4')

def lacc(mdot, rstar=u.R_sun, mstar=u.M_sun):
    return (mdot / 1.25 * constants.G * mstar / rstar).to(u.L_sun)

def log_pab(lacc):
    # log L_acc = a log L_pab + b
    # a = 1.06
    # b = 2.76
    # log_pab = (log L_acc - b) / a
    return 1/1.06 * np.log10(lacc/u.L_sun) - 2.76/1.06

def log_paa(lacc):
    # L_paa = L_pab * (L_Hb/L_pab) * (L_paa/L_hb)
    # L_Hb / L_pab = (L_hgamma / L_pab) * (L_hbeta/L_Hgamma) = (1/0.347) * (1/0.469)
    # L_paa / L_hb = 0.336
    # L_paa = L_pab * 0.336 / (0.347 * 0.469)
    # L_paa = 2.06 * L_pab
    # log L_paa = log 2.06 + log Pab
    return log_pab(lacc) + np.log10(0.336/(0.347*0.469))

def L_paa(lacc):
    return 10**log_paa(lacc) * u.L_sun



tblfn = 'coelho14_model_paa.fits'

if not os.path.exists(tblfn):
    data = []
    for fn in ProgressBar(glob.glob("s_coelho14_sed/*fits")):
        fh = fits.open(fn)
        header = fh[0].header
        sp = lower_dimensional_structures.OneDSpectrum.from_hdu(fh)
        #sp = specutils.Spectrum1D(data=fh[0].data, wcs=wcs.WCS(header), meta={'header': header})
        x = 10**sp.spectral_axis * u.AA

        # sp has units erg/s/cm^2, emission from stellar surface
        # let's just assume R_sun for now
        surfarea = 4*np.pi*constants.R_sun**2
        unit = u.erg/u.s/u.cm**2
        f_paa = sp[((18756-25)*u.AA < x) & (x < (18756+25)*u.AA)].sum()*unit * surfarea
        f_paacl = sp[(18620*u.AA < x) & (x < 18720*u.AA)].sum()*unit * surfarea
        f_paach = sp[(18790*u.AA < x) & (x < 18890*u.AA)].sum()*unit * surfarea
        L_paa_4 = L_paa(lacc(1e-4*u.M_sun/u.yr))
        L_paa_6 = L_paa(lacc(1e-6*u.M_sun/u.yr))
        L_paa_8 = L_paa(lacc(1e-8*u.M_sun/u.yr))
        f_paa_accr_m4 = L_paa_4 + f_paa
        f_paa_accr_m6 = L_paa_6 + f_paa
        f_paa_accr_m8 = L_paa_8 + f_paa


        row = {'fn': fn,
               'teff':header['TEFF'],
               'logg': header['LOG_G'],
               'afe': header['AFE'],
               'feh': header['FEH'],
               'z':header['Z'],
               'paa':f_paa.to(u.L_sun).value,
               'paa_plus_accr_1e-4': f_paa_accr_m4.to(u.L_sun).value,
               'paa_plus_accr_1e-6': f_paa_accr_m6.to(u.L_sun).value,
               'paa_plus_accr_1e-8': f_paa_accr_m8.to(u.L_sun).value,
               'paac_l': f_paacl.to(u.L_sun).value,
               'paac_h': f_paach.to(u.L_sun).value}

        for band, bandtbl in zip(('J','H','K', '3.6', '4.5', '5.8', '8.0'),
                                 (jcurve, hcurve, kcurve, s36curve, s45curve, s58curve, s80curve)):
            trans = np.interp(x, bandtbl['Wavelength'].quantity, bandtbl['Transmission'])
            row[band] = ((sp*unit*surfarea).to(u.L_sun).value*trans).sum() / trans.sum()

        data.append(row)


    tbl = table.Table(data)
    tbl.add_column(col=-2.5*(np.log10(tbl['K'])), name='mK')
    tbl.add_column(col=-2.5*(np.log10(tbl['H'])), name='mH')
    tbl.add_column(col=-2.5*(np.log10(tbl['J'])), name='mJ')
    tbl.add_column(col=-2.5*(np.log10(tbl['3.6'])), name='m3.6')
    tbl.add_column(col=-2.5*(np.log10(tbl['4.5'])), name='m4.5')
    tbl.add_column(col=-2.5*(np.log10(tbl['5.8'])), name='m5.8')
    tbl.add_column(col=-2.5*(np.log10(tbl['8.0'])), name='m8.0')
    tbl.add_column(col=tbl['mH'] - tbl['mK'], name='H-K')
    tbl.add_column(col=tbl['mJ'] - tbl['mH'], name='J-H')
    tbl.add_column(col=tbl['m3.6'] - tbl['mK'], name='3.6-K')
    tbl.add_column(col=tbl['m3.6'] - tbl['m4.5'], name='3.6-4.5')
    tbl.add_column(col=tbl['m3.6'] - tbl['m5.8'], name='3.6-5.8')
    tbl.add_column(col=tbl['m4.5'] - tbl['m8.0'], name='4.5-8.0')
    tbl.add_column(col=tbl['m3.6'] - tbl['m8.0'], name='3.6-8.0')
    tbl.add_column(col=tbl['m5.8'] - tbl['m8.0'], name='5.8-8.0')
    tbl.add_column(col=tbl['mK'] - tbl['m8.0'], name='K-8.0')
    tbl.add_column(col=-2.5*(np.log10(tbl['paa'])), name='mag_paa')
    tbl.add_column(col=-(tbl['paa'] - (tbl['paac_l'] + tbl['paac_h'])/4), name='cont_m_paa')
    tbl.add_column(col=-2.5*(np.log10(tbl['cont_m_paa'])), name='mag_cont_m_paa')
    tbl.add_column(col=-2.5*(np.log10(tbl['paac_l'])), name='mag_paac_l')
    tbl.add_column(col=-2.5*(np.log10(tbl['paac_h'])), name='mag_paac_h')
    tbl.add_column(col=tbl['mag_paac_h']-tbl['mag_paac_l'], name='paach-paacl')
    tbl.add_column(col=tbl['mag_paa'] - np.log10(tbl['paac_h']), name='mag_paa-h')
    tbl.add_column(col=tbl['mag_paa'] - np.log10(tbl['paac_l']), name='mag_paa-l')
    tbl.add_column(col=tbl['mag_paa'] - np.log10(tbl['K']), name='mag_paa-K')
    tbl.add_column(col=tbl['mag_paa'] - np.log10(tbl['H']), name='mag_paa-H')
    tbl.add_column(col=tbl['m3.6'] - tbl['mag_paa'], name='3.6-paa')
    tbl.add_column(col=tbl['m8.0'] - tbl['mag_paa'], name='8.0-paa')
    tbl.add_column(col=tbl['m4.5'] - tbl['mag_paa'], name='4.5-paa')
    tbl.add_column(col=tbl['m5.8'] - tbl['mag_paa'], name='5.8-paa')
    for mdot in (4,6,8):
        tbl.add_column(col=-2.5*(np.log10(tbl[f'paa_plus_accr_1e-{mdot}'])), name=f'mag_paa_1e-{mdot}')
        for ir in ('3.6', '4.5', '5.8', '8.0', 'J', 'H', 'K'):
            tbl.add_column(col=tbl[f'm{ir}'] - tbl[f'mag_paa_1e-{mdot}'], name=f'{ir}-paa_accr_1e-{mdot}')

    tbl.write(tblfn, overwrite=True)
else:
    tbl = table.Table.read(tblfn)
