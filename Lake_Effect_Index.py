import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def les_index(tS, t850, rh, tcc, tp, u850, v850):
    def get_val(obj):
        if hasattr(obj, 'values'):
            return obj.values.squeeze()
        return np.squeeze(obj)

    # 1. Get raw values
    tS_v = get_val(tS)
    t850_v = get_val(t850)
    rh_v = get_val(rh)
    tcc_v = get_val(tcc)
    tp_v = get_val(tp)
    u_v = get_val(u850)
    v_v = get_val(v850)

    # 2. Pixel-Level Safety Mask
    core_vars = [tS_v, t850_v, rh_v, tcc_v, tp_v]
    nan_count = np.sum([np.isnan(v) for v in core_vars], axis=0)
    missing_data_mask = (nan_count >= 2)

    # 3. Clean data
    tS = np.nan_to_num(tS_v)
    t850 = np.nan_to_num(t850_v)
    rh = np.nan_to_num(rh_v)
    tcc = np.nan_to_num(tcc_v)
    tp = np.nan_to_num(tp_v, nan=0)
    u850 = np.nan_to_num(u_v, nan=0)
    v850 = np.nan_to_num(v_v, nan=0)

    # 4. Thermodynamics
    tS_f = (tS - 273.15) * 9/5 + 32
    t850_f = (t850 - 273.15) * 9/5 + 32
    delta_T = tS_f - t850_f

    # Wind direction
    wind_dir = (270 - np.degrees(np.arctan2(v850, u850))) % 360
    f_fetch = np.where((wind_dir >= 220) & (wind_dir <= 290), 1, 0)

    # Scaling
    f_dT = np.clip(((delta_T - 15) / 2), 0, 1)
    f_RH = np.clip(((rh - 75) / 20), 0, 1)
    f_P = np.clip((tp / 3.0), 0, 1)

    f_T = np.zeros_like(tS_f)
    f_T = np.where((tS_f >= 10) & (tS_f <= 30), 1, f_T)
    f_T = np.where((tS_f < 37) & (tS_f > 30), ((36 - tS_f) / 8), f_T)
    f_T = np.where((tS_f < 10) & (tS_f > -4), ((tS_f + 4) / 14), f_T)

    # Final index
    base_index = 100 * (0.1*f_dT + 0.4*f_RH + 0.1*f_T + 0.4*f_P)
    index = base_index

    index[missing_data_mask] = 0

    return index


def main():
    ds_run = xr.open_dataset("GFS_2022111700.nc")

    # Initialization time
    initial_time = ds_run['tS'].isel(valid_time=0)
    initial_utc = pd.to_datetime(initial_time.valid_time.values)
    initial_est = initial_utc.tz_localize('UTC').tz_convert('US/Eastern')
    initial_est_str = initial_est.strftime('%b %d, %Y - %I:%M %p EST')

    for vt in range(len(ds_run.valid_time)):

        les_raw = les_index(
            ds_run['tS'].isel(valid_time=vt),
            ds_run['t'].sel(isobaricInhPa=850).isel(valid_time=vt),
            ds_run['r'].sel(isobaricInhPa=850).isel(valid_time=vt),
            ds_run['tcc'].isel(valid_time=vt),  
            ds_run['tp'].isel(valid_time=vt),
            ds_run['u'].sel(isobaricInhPa=850).isel(valid_time=vt),
            ds_run['v'].sel(isobaricInhPa=850).isel(valid_time=vt)
        )

        les_t = ds_run['tS'].isel(valid_time=vt).copy(data=les_raw)

        # Plot
        fig = plt.figure(figsize=(12, 10), dpi=144)
        ax = plt.axes(projection=ccrs.LambertConformal(
            central_longitude=-72.5, central_latitude=42.0))

        res = '10m'
        ax.add_feature(cfeature.LAND.with_scale(res), facecolor='#fdfdfd')
        ax.add_feature(cfeature.LAKES.with_scale(res), edgecolor='black',
                       facecolor='#d1e9ff', linewidth=1.2)
        ax.add_feature(cfeature.STATES.with_scale(res), edgecolor='#666666',
                       linewidth=0.8, linestyle=':')
        ax.add_feature(cfeature.COASTLINE.with_scale(res),
                       edgecolor='black', linewidth=1.5)

        plot = ax.contourf(
            les_t.longitude,
            les_t.latitude,
            les_t.values,
            levels=[0, 25, 50, 75, 100],
            cmap=plt.get_cmap('Blues', 5),
            transform=ccrs.PlateCarree(),
            extend='max',
            alpha=0.85
        )

        plt.colorbar(plot, ax=ax, orientation='horizontal',
                     pad=0.05, shrink=0.6)

        # Streamlines
        ax.streamplot(
            ds_run.longitude.values,
            ds_run.latitude.values,
            ds_run['u'].sel(isobaricInhPa=850).isel(valid_time=vt).values,
            ds_run['v'].sel(isobaricInhPa=850).isel(valid_time=vt).values,
            transform=ccrs.PlateCarree(),
            density=1.0,
            color='k'
        )

        ax.set_extent([-82.5, -72.5, 41.0, 45.0])

        # Time formatting
        valid_time_utc = pd.to_datetime(les_t.valid_time.values)
        valid_time_est = valid_time_utc.tz_localize('UTC').tz_convert('US/Eastern')
        time_str = valid_time_est.strftime('%b %d, %Y - %I:%M %p EST')

        plt.title(
            f"LESPI\nInit: {initial_est_str}\nValid: {time_str}",
            fontsize=14
        )

        filename = f"threat_{vt:03d}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        print(f"Saved {filename}")


if __name__ == "__main__":
    main()