# ==============================================================================
# 🔬 PIPELINE COMPLETA ALMA × IRAC — Tutti i clump, bande I3 + I4
# ==============================================================================

# --- IMPORT ---
import numpy as np
import os
import pandas as pd
import warnings
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip, mad_std
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
from reproject import reproject_interp
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                 ApertureStats, aperture_photometry)

warnings.filterwarnings('ignore', category=AstropyWarning)

# ==============================================================================
# 📁 PATH
# ==============================================================================
# Per Colab:
# alma_file      = '/content/drive/MyDrive/AlmaIrac/7MTM2TM1_Core_catalogue_out_official_v3_run6Apr23+16Jan24_sn-5_may24_clump_cat.txt'
# alma_base_path = '/content/drive/MyDrive/AlmaIrac/7MTM2TM1'
# irac_base_path = '/content/drive/MyDrive/AlmaIrac/Infrarosso/IR'
# irac_cat_path  = '/content/drive/MyDrive/AlmaIrac/Infrarosso/Catalogs/SourceCatalogue_glimpse_s07_ar'
# output_dir     = '/content/drive/MyDrive/AlmaIrac/Risultati_finali'

# Per Windows con Google Drive Desktop (decommentare e adattare):
base = 'G:/Il mio Drive/AlmaIrac'
alma_file      = 'G:/Il mio Drive/AlmaIrac/7MTM2TM1_Core_catalogue_out_official_v3_run6Apr23+16Jan24_sn-5_may24_clump_cat.txt'
alma_base_path = 'G:/.shortcut-targets-by-id/1mb52ggJU01t6_Z5Ek4LwajVsrsbc4LQv/7MTM2TM1'
irac_base_path = 'G:/.shortcut-targets-by-id/1NkSwpxs1l_4M32lz1vq4NoXzjBTCy2w9/Infrarosso/IR'
irac_cat_path  = 'G:/.shortcut-targets-by-id/1NkSwpxs1l_4M32lz1vq4NoXzjBTCy2w9/Infrarosso/Catalogs/SourceCatalogue_glimpse_s07_ar'
output_dir     = 'G:/Il mio Drive/AlmaIrac/Risultati_finali'

os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# ⚙️ PARAMETRI
# ==============================================================================
MAX_SEP_ARCSEC = 2.0          # soglia matching ALMA-IRAC
R_PHOT_ARCSEC  = 3.0          # raggio apertura fotometria forzata
R_ANN_IN_ARCSEC  = 3.0        # annulus interno
R_ANN_OUT_ARCSEC = 7.0        # annulus esterno
SIGCLIP_SIGMA  = 3.0

# Aperture correction per banda 
AP_CORR = {'I3': 1.290, 'I4': 1.473}

# Colonna flusso nel catalogo GLIMPSE per banda
FLUX_COL = {'I3': 'f5_8', 'I4': 'f8_0'}

BANDE = ['I3', 'I4']

print("Caricamento catalogo ALMA...")
df_alma_cat = pd.read_csv(alma_file, sep=r'\s+', skiprows=3, header=None)
# Converti in structured array compatibile col resto del codice
data = np.core.records.fromarrays(df_alma_cat.values.T,
                                   names=[f'f{i}' for i in range(df_alma_cat.shape[1])])
all_clumps = np.unique(data['f1'].astype(str))
print(f"Clump totali: {len(all_clumps)}")

# Pre-carica lista file ALMA
alma_all_files = os.listdir(alma_base_path)
print(f"File ALMA: {len(alma_all_files)}")

sigclip = SigmaClip(sigma=SIGCLIP_SIGMA)

# ==============================================================================
# 📋 LISTE RISULTATI
# ==============================================================================
all_fov_results = []       # flussi integrati sul FOV
all_source_results = []    # flussi per singola sorgente
error_log = []

n_tot = len(all_clumps)
print(f"\nInizio elaborazione su {n_tot} clump...\n")

# ==============================================================================
# 🔁 LOOP PRINCIPALE
# ==============================================================================
#for ic, target_clump in enumerate(all_clumps[:5]):
for ic, target_clump in enumerate(['47184']):

    if (ic + 1) % 20 == 0 or ic == 0 or ic == n_tot - 1:
        print(f"  [{ic+1}/{n_tot}] ({100*(ic+1)/n_tot:.0f}%) — clump {target_clump}")

    # ==================================================================
    # 📡 A. APRI ALMA
    # ==================================================================
    alma_fits_list = [
        f for f in alma_all_files
        if f.startswith(f'{target_clump}_cont_') and f.endswith('.pbcor.fits')
    ]
    if len(alma_fits_list) == 0:
        error_log.append({'Clump_ID': target_clump, 'Error': 'No ALMA pbcor'})
        continue

    try:
        with fits.open(os.path.join(alma_base_path, alma_fits_list[0])) as hdul:
            alma_data_img = hdul[0].data.squeeze()
            alma_header   = hdul[0].header
    except Exception as e:
        error_log.append({'Clump_ID': target_clump, 'Error': f'ALMA open: {e}'})
        continue

    # --- WCS 2D + geometria campo ---
    alma_wcs_2d    = WCS(alma_header, naxis=2)
    alma_header_2d = alma_wcs_2d.to_header()
    alma_header_2d['NAXIS']  = 2
    alma_header_2d['NAXIS1'] = alma_data_img.shape[1]
    alma_header_2d['NAXIS2'] = alma_data_img.shape[0]

    ny, nx = alma_data_img.shape
    cdelt  = abs(alma_header.get('CDELT2', 2e-5))
    cx     = alma_header_2d['CRPIX1'] - 1
    cy     = alma_header_2d['CRPIX2'] - 1

    yy, xx = np.mgrid[0:ny, 0:nx]
    dist_from_c = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    valid_alma = np.isfinite(alma_data_img)

    if valid_alma.sum() == 0:
        error_log.append({'Clump_ID': target_clump, 'Error': 'ALMA all NaN'})
        continue

    alma_radius_pix = dist_from_c[valid_alma].max()
    alma_mask = dist_from_c <= alma_radius_pix

    pixel_area_sr      = (cdelt * np.pi / 180)**2
    pixel_scale_arcsec = cdelt * 3600

    # ==================================================================
    # 📡 B. FLUSSO INTEGRATO ALMA SUL FOV
    # ==================================================================
    bmaj = alma_header.get('BMAJ', None)
    bmin = alma_header.get('BMIN', None)

    if bmaj is not None and bmin is not None:
        beam_area_sr = (np.pi / (4 * np.log(2))) * (bmaj * np.pi/180) * (bmin * np.pi/180)
        pix_per_beam = beam_area_sr / pixel_area_sr
        conversion   = pixel_area_sr / beam_area_sr
    else:
        conversion = None

    alma_in_field = alma_data_img[alma_mask & valid_alma]
    alma_rms = mad_std(alma_in_field, ignore_nan=True)
    alma_sum = np.nansum(alma_in_field)

    if conversion is not None:
        flux_alma_fov_mJy       = alma_sum * conversion * 1e3
        above_3sigma = alma_in_field[alma_in_field > 3 * alma_rms]
        flux_alma_fov_3sig_mJy  = np.sum(above_3sigma) * conversion * 1e3 if len(above_3sigma) > 0 else 0.0
    else:
        flux_alma_fov_mJy      = np.nan
        flux_alma_fov_3sig_mJy = np.nan

    # ==================================================================
    # 📡 C. INFO SORGENTI ALMA DAL CATALOGO
    # ==================================================================
    mask_clump = data['f1'].astype(str) == target_clump
    clump_data = data[mask_clump]

    ra_alma       = clump_data['f4'].astype(float)
    dec_alma      = clump_data['f5'].astype(float)
    alma_ids      = clump_data['f0'].astype(str).tolist()
    flux_alma_mJy = clump_data['f17'].astype(float) * 1e3   # F_INT Jy → mJy

    coords_alma = SkyCoord(ra=ra_alma*u.deg, dec=dec_alma*u.deg, frame='icrs')
    x_alma_pix, y_alma_pix = alma_wcs_2d.world_to_pixel(coords_alma)

    n_sources = len(alma_ids)

    # ==================================================================
    # 📡 D. CATALOGO GLIMPSE
    # ==================================================================
    cat_file = os.path.join(irac_cat_path,
                            f'{target_clump}_IRsources_glimpse_s07_ar.txt')
    has_glimpse_cat = os.path.exists(cat_file)
   
    if has_glimpse_cat:
        try:
            cat_glimpse = Table.read(cat_file, format='ipac')
            cat_glimpse = cat_glimpse.filled(np.nan)

            coords_cat = SkyCoord(ra=cat_glimpse['ra'], dec=cat_glimpse['dec'],
                                  unit='deg', frame='icrs')
            xs_cat, ys_cat = alma_wcs_2d.world_to_pixel(coords_cat)

            # Filtra solo sorgenti dentro il FOV ALMA
            dist_cat = np.sqrt((xs_cat - cx)**2 + (ys_cat - cy)**2)
            inside_fov = dist_cat <= alma_radius_pix
            

            cat_finale   = cat_glimpse[inside_fov]
            xs_finale    = xs_cat[inside_fov]
            ys_finale    = ys_cat[inside_fov]

            # Short IDs
            short_ids = [str(name).strip().split('.')[-1]
                         for name in cat_finale['designation']]
        except Exception as e:
            has_glimpse_cat = False
            error_log.append({'Clump_ID': target_clump,
                              'Error': f'GLIMPSE cat: {e}'})

    # ==================================================================
    # 🔁 E. LOOP SULLE BANDE IRAC
    # ==================================================================
    clump_folder = os.path.join(irac_base_path, target_clump)

    for banda in BANDE:

        flux_col = FLUX_COL[banda]
        ap_corr  = AP_CORR[banda]

        # --- E1. Cerca e apri FITS IRAC ---
        has_irac = False
        irac_reproj = None

        if os.path.exists(clump_folder):
            fits_files = [
                f for f in os.listdir(clump_folder)
                if f.endswith('.fits') and f'_{banda}_' in f and 'asec360' in f
            ]
            if len(fits_files) > 0:
                try:
                    with fits.open(os.path.join(clump_folder, fits_files[0])) as hdul:
                        irac_data   = hdul[0].data.squeeze()
                        irac_header = hdul[0].header

                    irac_reproj, footprint = reproject_interp(
                        (irac_data, irac_header), alma_header_2d, order='bilinear'
                    )
                    irac_masked = np.where(alma_mask, irac_reproj, np.nan)
                    has_irac = True
                    
                except Exception as e:
                    error_log.append({'Clump_ID': target_clump,
                                      'Error': f'{banda} reproject: {e}'})

# --- E2. Flusso integrato IRAC sul FOV ---
        if has_irac:
            irac_valid = irac_masked[np.isfinite(irac_masked)]
            if len(irac_valid) > 0:
                outer = (~alma_mask) & np.isfinite(irac_reproj)
                if outer.sum() > 100:
                    bkg = np.nanmedian(irac_reproj[outer])
                    n_pix_in = np.isfinite(irac_masked).sum()
                    flux_irac_fov_mJy = (np.nansum(irac_masked) - bkg * n_pix_in) * pixel_area_sr * 1e6 * 1e3
                else:
                    flux_irac_fov_mJy = np.nansum(irac_masked) * pixel_area_sr * 1e6 * 1e3
            else:
                flux_irac_fov_mJy = np.nan
        else:
            flux_irac_fov_mJy = np.nan

        # Salva nel dizionario temporaneo FOV
        if banda == BANDE[0]:
            fov_temp = {
                'Clump_ID':         target_clump,
                'F_ALMA_FOV_mJy':   round(flux_alma_fov_mJy, 4) if np.isfinite(flux_alma_fov_mJy) else np.nan,
                'F_ALMA_3sig_mJy':  round(flux_alma_fov_3sig_mJy, 4) if np.isfinite(flux_alma_fov_3sig_mJy) else np.nan,
                'DIST_kpc':         float(clump_data['f21'][0]),
                'Lclump_Lsun':      float(clump_data['f22'][0]),
                'Mclump_Msun':      float(clump_data['f23'][0]),
                'Tclump_K':         float(clump_data['f24'][0]),
                'Surfd_clump':      float(clump_data['f25'][0]),
                'L_over_M':         float(clump_data['f31'][0]),
                'EVOL_FLAG':        int(clump_data['f32'][0]),
                'N_FRAG':           int(clump_data['f20'][0]),
            }

        fov_temp[f'F_IRAC_{banda}_FOV_mJy'] = round(flux_irac_fov_mJy, 4) if np.isfinite(flux_irac_fov_mJy) else np.nan

        # Dopo ultima banda: calcola rapporti e appendi
        if banda == BANDE[-1]:
            f_i3 = fov_temp.get('F_IRAC_I3_FOV_mJy', np.nan)
            f_i4 = fov_temp.get('F_IRAC_I4_FOV_mJy', np.nan)
            f_mm = fov_temp['F_ALMA_FOV_mJy']

            fov_temp['Ratio_mm_I3_FOV'] = round(f_mm / f_i3, 6) if (np.isfinite(f_i3) and f_i3 > 0) else np.nan
            fov_temp['Ratio_mm_I4_FOV'] = round(f_mm / f_i4, 6) if (np.isfinite(f_i4) and f_i4 > 0) else np.nan
            fov_temp['Ratio_I3_I4_FOV'] = round(f_i3 / f_i4, 4) if (np.isfinite(f_i3) and np.isfinite(f_i4) and f_i4 > 0) else np.nan

            all_fov_results.append(fov_temp)

        # --- E3. Maschera sorgenti per fotometria forzata ---
        source_mask = np.zeros((ny, nx), dtype=bool)
        r_mask_pix = 2.5 / pixel_scale_arcsec   # 2.5" in pixel

        # Maschera sorgenti ALMA
        for xi, yi in zip(x_alma_pix, y_alma_pix):
            d = np.sqrt((xx - xi)**2 + (yy - yi)**2)
            source_mask |= (d <= r_mask_pix)

        # Maschera sorgenti GLIMPSE dentro FOV
        if has_glimpse_cat and len(cat_finale) > 0:
            for xi, yi in zip(xs_finale, ys_finale):
                d = np.sqrt((xx - xi)**2 + (yy - yi)**2)
                source_mask |= (d <= r_mask_pix)

# ==============================================================
        # 🎯 E4. Match + fotometria 
        # ==============================================================
        r_phot_pix    = R_PHOT_ARCSEC / pixel_scale_arcsec
        r_ann_in_pix  = R_ANN_IN_ARCSEC / pixel_scale_arcsec
        r_ann_out_pix = R_ANN_OUT_ARCSEC / pixel_scale_arcsec
        to_mJy = pixel_area_sr * 1e6 * 1e3

        # --- Prima banda: match posizionale + inizializza ---
        if banda == BANDE[0]:
            source_temp = {}

            for i in range(n_sources):
                nome_alma = alma_ids[i]

                irac_id = 'N/A'
                sep = np.nan
                matched_idx = None

                if has_glimpse_cat and len(cat_finale) > 0:
                    # Filtra: sorgenti con flusso valido in almeno una banda
                    has_any_flux = np.zeros(len(cat_finale), dtype=bool)
                    for b in BANDE:
                        col = FLUX_COL[b]
                        has_any_flux |= (~np.isnan(np.array(cat_finale[col]))) & (np.array(cat_finale[col]) > 0)

                    if has_any_flux.sum() > 0:
                        xs_valid = xs_finale[has_any_flux]
                        ys_valid = ys_finale[has_any_flux]
                        valid_idx = np.where(has_any_flux)[0]

                        dist_pix = np.sqrt((xs_valid - x_alma_pix[i])**2 +
                                           (ys_valid - y_alma_pix[i])**2)
                        dist_arcsec = dist_pix * pixel_scale_arcsec
                        idx_closest = np.argmin(dist_arcsec)
                        min_dist = dist_arcsec[idx_closest]

                        if min_dist <= MAX_SEP_ARCSEC:
                            matched_idx = valid_idx[idx_closest]
                            irac_id = short_ids[matched_idx]
                            sep = round(float(min_dist), 3)

                source_temp[nome_alma] = {
                    'Clump_ID': target_clump,
                    'ALMA_ID': nome_alma,
                    'F_ALMA_mJy': round(flux_alma_mJy[i], 4),
                    'SN': round(float(clump_data['f9'][i]), 2),           # S/N
                    'Tcore_K': float(clump_data['f26'][i]),                # temperatura core
                    'Mcore_Msun': float(clump_data['f27'][i]),             # massa core
                    'Dcore_AU': float(clump_data['f28'][i]),               # diametro core
                    'Surfd_core': float(clump_data['f46'][i]),             # surface density core
                    'nH2_cm3': float(clump_data['f47'][i]),                # densità H2
                # ... match info ...
                    'IRAC_ID': irac_id,
                    'Sep_arcsec': sep,
                    'matched_idx': matched_idx,
                }

        # --- Per ogni banda: estrai flusso (catalogo o forzata) ---
        for i in range(n_sources):
            nome_alma = alma_ids[i]
            matched_idx = source_temp[nome_alma]['matched_idx']

            f_ir = np.nan
            ir_source = 'No_IRAC'

            # Se c'è match posizionale, prova a prendere il flusso da catalogo
            if matched_idx is not None and has_glimpse_cat:
                f_cat_val = cat_finale[flux_col][matched_idx]
                if not np.isnan(f_cat_val) and f_cat_val > 0:
                    f_ir = float(f_cat_val)
                    ir_source = 'Catalogo'

            # Se no flusso da catalogo ma c'è immagine IRAC → forzata
            if ir_source != 'Catalogo' and has_irac:
                try:
                    ap  = CircularAperture((x_alma_pix[i], y_alma_pix[i]), r=r_phot_pix)
                    ann = CircularAnnulus((x_alma_pix[i], y_alma_pix[i]),
                                          r_in=r_ann_in_pix, r_out=r_ann_out_pix)

                    phot = aperture_photometry(irac_reproj, ap)
                    raw = phot['aperture_sum'][0]

                    stats = ApertureStats(irac_reproj, ann,
                                          sigma_clip=sigclip, mask=source_mask)
                    if not np.isnan(stats.median):
                        f_ir = float((raw - stats.median * ap.area) * to_mJy) * ap_corr
                    else:
                        stats_nomask = ApertureStats(irac_reproj, ann, sigma_clip=sigclip)
                        f_ir = float((raw - stats_nomask.median * ap.area) * to_mJy) * ap_corr

                    ir_source = 'Forzata'
                except:
                    f_ir = np.nan
                    ir_source = 'Error'

            source_temp[nome_alma][f'F_IRAC_{banda}_mJy'] = round(f_ir, 4) if np.isfinite(f_ir) else np.nan
            source_temp[nome_alma][f'IR_Source_{banda}'] = ir_source

        # --- Dopo ultima banda: calcola rapporti e appendi ---
        if banda == BANDE[-1]:
            for nome_alma in source_temp:
                row = source_temp[nome_alma]
                del row['matched_idx']   # rimuovi campo interno

                f_i3 = row.get('F_IRAC_I3_mJy', np.nan)
                f_i4 = row.get('F_IRAC_I4_mJy', np.nan)
                f_mm = row['F_ALMA_mJy']

                row['Ratio_mm_I3'] = round(f_mm / f_i3, 4) if (np.isfinite(f_i3) and f_i3 > 0) else np.nan
                row['Ratio_mm_I4'] = round(f_mm / f_i4, 4) if (np.isfinite(f_i4) and f_i4 > 0) else np.nan
                row['Ratio_I3_I4'] = round(f_i3 / f_i4, 4) if (np.isfinite(f_i3) and np.isfinite(f_i4) and f_i4 > 0) else np.nan

                all_source_results.append(row)

    # Libera memoria
    del alma_data_img, alma_mask, dist_from_c, valid_alma, yy, xx
    if irac_reproj is not None:
        del irac_reproj, irac_masked

# ==============================================================================
# 💾 SALVATAGGIO RISULTATI
# ==============================================================================
print(f"\n{'='*60}")
print(f"  SALVATAGGIO RISULTATI")
print(f"{'='*60}")

# FOV fluxes
df_fov = pd.DataFrame(all_fov_results)
fov_path = os.path.join(output_dir, 'fov_fluxes.csv')
df_fov.to_csv(fov_path, index=False)
print(f"  FOV fluxes: {fov_path} ({len(df_fov)} righe)")

# Source results
df_src = pd.DataFrame(all_source_results)
src_path = os.path.join(output_dir, 'source_results.csv')
df_src.to_csv(src_path, index=False)
print(f"  Source results: {src_path} ({len(df_src)} righe)")

# Error log
if error_log:
    df_err = pd.DataFrame(error_log)
    err_path = os.path.join(output_dir, 'error_log.csv')
    df_err.to_csv(err_path, index=False)
    print(f"  Error log: {err_path} ({len(df_err)} errori)")

# ==============================================================================
# 📊 RIEPILOGO FINALE
# ==============================================================================
print(f"\n{'='*60}")
print(f"  RIEPILOGO FINALE")
print(f"{'='*60}")
print(f"  Clump processati          : {df_fov['Clump_ID'].nunique()}")
df_src = pd.DataFrame(all_source_results)
print(f"  Sorgenti ALMA totali      : {len(df_src)}")
for banda in BANDE:
    n_cat = (df_src[f'IR_Source_{banda}'] == 'Catalogo').sum()
    n_for = (df_src[f'IR_Source_{banda}'] == 'Forzata').sum()
    n_no  = (df_src[f'IR_Source_{banda}'] == 'No_IRAC').sum()
    print(f"  --- Banda {banda} ---")
    print(f"    Match catalogo GLIMPSE  : {n_cat}")
    print(f"    Fotometria forzata      : {n_for}")
    print(f"    Senza IRAC              : {n_no}")
print(f"  Errori                    : {len(error_log)}")
print(f"{'='*60}")
print("\n✅ PIPELINE COMPLETATA")
