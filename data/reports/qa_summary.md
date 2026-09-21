# QA Summary — Raw Data Audit

Generated from 15 raw files, 140640 total rows, 247 data columns.

## Per-file timing audit

| source_file                         |   n_rows |   t_start_s |   t_end_s |   duration_s | is_monotonic   |   n_duplicate_timestamps |   effective_freq_hz_median |   dt_mean_s |   dt_std_s |   n_gaps_gt_3x_median_dt |   max_gap_s |   n_nan_total |   n_inf_total |
|:------------------------------------|---------:|------------:|----------:|-------------:|:---------------|-------------------------:|---------------------------:|------------:|-----------:|-------------------------:|------------:|--------------:|--------------:|
| circuito-completo.csv               |    26489 |   0.0166667 |   443.142 |      443.125 | True           |                        0 |                         60 |   0.0167293 | 0.00409102 |                       25 |    0.35     |             0 |             0 |
| turbulencia05.csv                   |     7146 |   0.0166667 |   119.342 |      119.325 | True           |                        0 |                         60 |   0.0167005 | 0.00396204 |                        4 |    0.3      |             0 |             0 |
| turbulencia08.csv                   |     7894 |   0.0166667 |   131.817 |      131.8   | True           |                        0 |                         60 |   0.0166983 | 0.00385843 |                        4 |    0.308333 |             0 |             0 |
| voo-2-com-16-nos.csv                |     7855 |   0.0166667 |   131.333 |      131.317 | True           |                        0 |                         60 |   0.0167197 | 0.00424908 |                        5 |    0.3      |             0 |             0 |
| voo-2-com-rajada.csv                |     7957 |   0.0166667 |   132.867 |      132.85  | True           |                        0 |                         60 |   0.0166981 | 0.00400333 |                        3 |    0.316667 |             0 |             0 |
| voo-3-com-16-nos.csv                |     7683 |   0.0166667 |   128.3   |      128.283 | True           |                        0 |                         60 |   0.0166992 | 0.00411166 |                        2 |    0.333333 |             0 |             0 |
| voo-com-16-nos.csv                  |     8603 |   0.0166667 |   143.617 |      143.6   | True           |                        0 |                         60 |   0.0166938 | 0.00347534 |                        2 |    0.283333 |             0 |             0 |
| voo-com-8-nos.csv                   |     6639 |   0.0166667 |   111.225 |      111.208 | True           |                        0 |                         60 |   0.0167533 | 0.00483073 |                        7 |    0.3      |             0 |             0 |
| voo-com-rajada-vindo-da-direita.csv |     8209 |   0.0166667 |   137.067 |      137.05  | True           |                        0 |                         60 |   0.0166971 | 0.00389822 |                        2 |    0.325    |             0 |             0 |
| voo-com-rajada.csv                  |     7119 |   0.0166667 |   119.05  |      119.033 | True           |                        0 |                         60 |   0.0167229 | 0.00415074 |                        3 |    0.3      |             0 |             0 |
| voo-condicoes-normais.csv           |     7362 |   0.0166667 |   122.917 |      122.9   | True           |                        0 |                         60 |   0.0166961 | 0.00371398 |                        3 |    0.283333 |             0 |             0 |
| voo-inicio-teste.csv                |    15589 |   0.0166667 |   261.033 |      261.017 | True           |                        0 |                         60 |   0.0167447 | 0.00425548 |                       13 |    0.291667 |             0 |             0 |
| voo-normal2.csv                     |     7281 |   0.0166667 |   121.8   |      121.783 | True           |                        0 |                         60 |   0.0167285 | 0.00467984 |                        4 |    0.3      |             0 |             0 |
| voo-normal3.csv                     |     7459 |   0.0166667 |   124.617 |      124.6   | True           |                        0 |                         60 |   0.0167069 | 0.00390306 |                        5 |    0.275    |             0 |             0 |
| voo2-com-8-nos.csv                  |     7355 |   0.0166667 |   122.783 |      122.767 | True           |                        0 |                         60 |   0.0166939 | 0.00362311 |                        2 |    0.291667 |             0 |             0 |


## Control range violations

No control-range violations detected across the curated control columns.


## Confirmed duplicate columns

| keep_column                                        | drop_column                                        | confirmed_duplicate   | formula                                               |   max_abs_error |   lag_samples |   tolerance_used | note                                                                                                                                |
|:---------------------------------------------------|:---------------------------------------------------|:----------------------|:------------------------------------------------------|----------------:|--------------:|-----------------:|:------------------------------------------------------------------------------------------------------------------------------------|
| /position[0]/altitude-agl-ft[0]                    | /position[0]/altitude-agl-m[0]                     | True                  | agl_m = agl_ft * 0.3048                               |       0.0050512 |             1 |        1.03581   | ft->m unit conversion (per-file consensus lag=1; worst file=voo-inicio-teste.csv; lags seen=[1])                                    |
| /velocities[0]/vertical-speed-fps[0]               | /velocities[0]/speed-down-fps[0]                   | True                  | speed_down_fps = -vertical_speed_fps                  |       0         |             0 |        0.0683399 | sign-flipped NED component (per-file consensus lag=0; worst file=circuito-completo.csv; lags seen=[0])                              |
| /controls[0]/engines[0]/engine[0]/throttle[0]      | /controls[0]/engines[0]/engine[1]/throttle[0]      | True                  | engine1_throttle ~= engine0_throttle (identity check) |       0         |             0 |        0.001     | C172P is single-engine; engine[1] may be an unused copy (per-file consensus lag=0; worst file=circuito-completo.csv; lags seen=[0]) |
| /surface-positions[0]/left-aileron-pos-norm[0]     | /surface-positions[0]/right-aileron-pos-norm[0]    | True                  | identity (byte-exact)                                 |       0         |           nan |      nan         | exact duplicate values                                                                                                              |
| /gear[0]/gear[0]/rollspeed-ms[0]                   | /engines[0]/engine[7]/rpm[0]                       | True                  | identity (byte-exact)                                 |       0         |           nan |      nan         | exact duplicate values                                                                                                              |
| /gear[0]/gear[1]/rollspeed-ms[0]                   | /engines[0]/engine[7]/n1[0]                        | True                  | identity (byte-exact)                                 |       0         |           nan |      nan         | exact duplicate values                                                                                                              |
| /gear[0]/gear[2]/rollspeed-ms[0]                   | /engines[0]/engine[7]/n2[0]                        | True                  | identity (byte-exact)                                 |       0         |           nan |      nan         | exact duplicate values                                                                                                              |
| /consumables[0]/fuel[0]/tank[0]/level-gal_us[0]    | /consumables[0]/fuel[0]/tank[1]/level-gal_us[0]    | True                  | identity (byte-exact)                                 |       0         |           nan |      nan         | exact duplicate values                                                                                                              |
| /controls[0]/engines[0]/engine[0]/throttle[0]      | /controls[0]/engines[0]/engine[1]/throttle[0]      | True                  | identity (byte-exact)                                 |       0         |           nan |      nan         | exact duplicate values                                                                                                              |
| /fdm[0]/jsbsim[0]/contact[0]/unit[4]/z-position[0] | /fdm[0]/jsbsim[0]/contact[0]/unit[5]/z-position[0] | True                  | identity (byte-exact)                                 |       0         |           nan |      nan         | exact duplicate values                                                                                                              |


## Candidate duplicate pairs NOT confirmed (kept, flagged)

| keep_column                      | drop_column                                 | confirmed_duplicate   | formula                                  |   max_abs_error |   lag_samples |   tolerance_used | note                                                                                                                              |
|:---------------------------------|:--------------------------------------------|:----------------------|:-----------------------------------------|----------------:|--------------:|-----------------:|:----------------------------------------------------------------------------------------------------------------------------------|
| /velocities[0]/airspeed-kt[0]    | /fdm[0]/jsbsim[0]/velocities[0]/vias-kts[0] | False                 | vias_kts ~= airspeed_kt (identity check) |        101.426  |            -2 |       0.0916933  | indicated vs calibrated airspeed (per-file consensus lag=-2; worst file=voo-normal2.csv; lags seen=[-2, 2])                       |
| /orientation[0]/side-slip-rad[0] | /orientation[0]/beta-deg[0]                 | False                 | beta_deg = degrees(side_slip_rad)        |         89.6182 |            -2 |       0.00163486 | deg vs rad sideslip representation (per-file consensus lag=-2; worst file=voo-com-rajada-vindo-da-direita.csv; lags seen=[-2, 2]) |


## Constant / near-constant columns (excluded)

133 columns out of 247 are constant or near-constant across the baseline-eligible files.


## Role distribution

| role                   |   count |
|:-----------------------|--------:|
| excluded               |     193 |
| qa_only                |      20 |
| neural_input           |      14 |
| flight_manager_only    |      11 |
| label                  |       4 |
| derived_feature_source |       3 |
| metadata               |       2 |