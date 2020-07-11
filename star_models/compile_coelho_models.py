import glob
from astropy import table
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
        tf.extractall()

tblfn = 'coelho14_model_paa.fits'

if not os.path.exists(tblfn):
    data = []
    for fn in glob.glob("s_coelho14_sed/*fits"):
        fh = fits.open(fn)
        header = fh[0].header
        sp = lower_dimensional_structures.OneDSpectrum.from_hdu(fh)
        x = 10**sp.spectral_axis
        data.append({'fn': fn,
                     'teff':header['TEFF'],
                     'logg': header['LOG_G'],
                     'afe': header['AFE'],
                     'feh': header['FEH'],
                     'z':header['Z'],
                     'paa':sp[(18756-25 < x) & (x < 18756+25)].sum(),
                     'paac_l':sp[(18620 < x) & (x < 18720)].sum(),
                     'paac_h': sp[(18790 < x) & (x < 18890)].sum()})



    tbl = table.Table(data)
    tbl.write(tblfn)
