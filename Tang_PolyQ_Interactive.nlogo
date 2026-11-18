extensions [ py ]

breed [ cells cell ]
breed [ oligomers oligomer ]
breed [ inclusions inclusion ]
breed [ proteasomes proteasome ]
breed [ ros-particles ros-particle ]

cells-own [
  cell-id
  has-inclusion?
  is-dead?
]

globals [
  sim-time-s
  sim-time-h
  is-unstable?
  status-msg
  curr-death-pct
  curr-pi-death-pct
  curr-p38-death-pct
  curr-inc-freq-pct
  curr-seq-agg
  curr-polyq-olig
  curr-misp-olig
  curr-free-prot
  curr-seq-prot
  curr-ros
  curr-p38-pct
  curr-mrfpu
  curr-lambda-max
  curr-dt-max
  t-seed-h
  t-exhaust-h
]

to setup
  clear-all
  set sim-time-s 0
  set sim-time-h 0
  set is-unstable? false
  set status-msg "Initializing Python Engine..."
  set t-seed-h -1
  set t-exhaust-h -1

  py:setup py:python
  py:run "import sys, os, numpy as np"
  py:run "if os.path.abspath('src') not in sys.path: sys.path.insert(0, os.path.abspath('src'))"
  py:run "from tang_polyq.model import load_default_model"
  py:run "from tang_polyq.scenarios import get_scenarios"
  py:run "from tang_polyq.stability import compute_jacobian"

  py:set "use_ma_dimer" mass-action-dimer?
  py:set "scen_key" scenario-mode
  py:set "dt_val" step-size-dt
  py:set "n_cells_val" n-ensemble-cells
  py:set "solver_mode" solver-type
  py:set "custom_override" use-custom-sliders?
  py:set "slider_kp38act" kp38act-slider
  py:set "slider_kproteff" kproteff-24h-slider
  py:set "slider_kgenROSp38" kgenROSp38-slider
  py:set "slider_kaggMisP" kaggMisP-slider

  (py:run
    "xml_file = 'data/BIOMD0000000285_url.xml'"
    "model = load_default_model(xml_path=xml_file, use_mass_action_dimer=bool(use_ma_dimer))"
    "scens = get_scenarios()"
    "scen = scens[scen_key]"
    "y_ode = scen.get_initial_state(model).copy()"
    "rng = np.random.default_rng(42)"
    "Y_mc = np.repeat(y_ode[:, None], int(n_cells_val), axis=1).astype(np.float64)"
    "alive_mask = np.ones(int(n_cells_val), dtype=bool)"
    "sp_map = model.species_map"
    "t_curr = 0.0"
  )

  setup-visual-world
  sync-from-python
  update-visuals
  set status-msg (word "Ready (" solver-type " | " scenario-mode ")")
  reset-ticks
end

to setup-visual-world
  ifelse solver-type = "Tau-Leap (Ensemble)" [
    ask patches [ set pcolor black ]
    let n-c n-ensemble-cells
    let cols ceiling sqrt n-c
    let spacing (world-width - 4) / cols
    let i 0
    while [ i < n-c ] [
      let r floor (i / cols)
      let c (i mod cols)
      let px (min-pxcor + 2 + c * spacing)
      let py (max-pycor - 2 - r * spacing)
      create-cells 1 [
        set cell-id i
        setxy px py
        set shape "circle"
        set size max list 0.6 (spacing * 0.85)
        set color sky
        set has-inclusion? false
        set is-dead? false
      ]
      set i i + 1
    ]
  ] [
    ask patches [
      let d distancexy 0 0
      ifelse d < 5 [
        set pcolor blue - 4
      ] [
        ifelse d < 15 [
          set pcolor sky - 4
        ] [
          set pcolor black
        ]
      ]
    ]
  ]
end

to go
  if sim-time-h >= 48.0 or is-unstable? [ stop ]

  py:set "dt_val" step-size-dt
  py:set "solver_mode" solver-type
  py:set "steps_per_tick" steps-per-tick
  py:set "custom_override" use-custom-sliders?
  py:set "slider_kp38act" kp38act-slider
  py:set "slider_kproteff" kproteff-24h-slider
  py:set "slider_kgenROSp38" kgenROSp38-slider
  py:set "slider_kaggMisP" kaggMisP-slider

  (py:run
    "dt_s = float(dt_val)"
    "n_sub = int(steps_per_tick)"
    "unstable_flag = False"
    "for _ in range(n_sub):"
    "    if t_curr >= 48.0 * 3600.0 - 1e-9: break"
    "    dt_step = min(dt_s, 48.0 * 3600.0 - t_curr)"
    "    t_sw = scen.switch_time"
    "    if t_sw is not None and t_curr < t_sw < t_curr + dt_step:"
    "        dt_step = t_sw - t_curr"
    "    p = scen.get_params_at(model, t_curr)"
    "    if custom_override:"
    "        p['kp38act'] = float(slider_kp38act)"
    "        p['kgenROSp38'] = float(slider_kgenROSp38)"
    "        p['kaggMisP'] = float(slider_kaggMisP)"
    "        if t_curr >= 24.0 * 3600.0:"
    "            p['kproteff'] = float(slider_kproteff)"
    "    if solver_mode == 'Euler (ODE)':"
    "        dy = model.rhs(t_curr, y_ode, p)"
    "        y_ode = y_ode + dt_step * dy"
    "    elif solver_mode == 'RK2 (ODE)':"
    "        k1 = model.rhs(t_curr, y_ode, p)"
    "        k2 = model.rhs(t_curr + dt_step, y_ode + dt_step * k1, p)"
    "        y_ode = y_ode + 0.5 * dt_step * (k1 + k2)"
    "    elif solver_mode == 'Tau-Leap (Ensemble)':"
    "        a = model.propensities(t_curr, Y_mc, p)"
    "        a[:, ~alive_mask] = 0.0"
    "        lam = np.minimum(a * dt_step, 1e7)"
    "        k_fires = rng.poisson(lam).astype(np.float64)"
    "        Y_mc = np.maximum(np.round(Y_mc + (model.stoich_matrix @ k_fires)), 0.0)"
    "        dead_now = (Y_mc[sp_map['PIdeath'], :] >= 1.0) | (Y_mc[sp_map['p38death'], :] >= 1.0)"
    "        alive_mask = ~dead_now"
    "        if np.any(alive_mask):"
    "            y_ode = np.mean(Y_mc[:, alive_mask], axis=1)"
    "        else:"
    "            y_ode = np.mean(Y_mc, axis=1)"
    "    t_curr += dt_step"
    "    if solver_mode != 'Tau-Leap (Ensemble)':"
    "        if (not np.all(np.isfinite(y_ode))) or np.max(np.abs(y_ode)) > 1e12 or np.min(y_ode) < -1.0:"
    "            unstable_flag = True"
    "            break"
  )

  set is-unstable? py:runresult "bool(unstable_flag)"
  if is-unstable? [
    set status-msg (word "UNSTABLE BLOWUP at t = " precision sim-time-h 2 "h (dt = " step-size-dt "s > dt_max)")
    ask patches with [ pcolor != black ] [ set pcolor red - 3 ]
    stop
  ]

  sync-from-python
  update-visuals
  tick
end

to sync-from-python
  (py:run
    "t_h = float(t_curr / 3600.0)"
    "polyq_o = float(sum(y_ode[sp_map[f'AggPolyQ{i}']] for i in range(1, 6)))"
    "misp_o = float(sum(y_ode[sp_map[f'AggP{i}']] for i in range(1, 6)))"
    "seq_v = float(y_ode[sp_map['SeqAggP']])"
    "prot_f = float(y_ode[sp_map['Proteasome']])"
    "prot_s = float(y_ode[sp_map['AggP_Proteasome']] + y_ode[sp_map['PolyQ_Proteasome']] + y_ode[sp_map['MisP_Proteasome']] + y_ode[sp_map['mRFPu_Proteasome']])"
    "ros_v = float(y_ode[sp_map['ROS']])"
    "mrfpu_v = float(y_ode[sp_map['mRFPu']])"
    "p38_v = float(y_ode[sp_map['p38']])"
    "p38p_v = float(y_ode[sp_map['p38_P']])"
    "p38_pct = float(100.0 * p38p_v / max(p38_v + p38p_v, 1e-12))"
    "if solver_mode == 'Tau-Leap (Ensemble)':"
    "    pi_d_arr = Y_mc[sp_map['PIdeath'], :] >= 1.0"
    "    p38_d_arr = Y_mc[sp_map['p38death'], :] >= 1.0"
    "    dead_arr = pi_d_arr | p38_d_arr"
    "    inc_arr = Y_mc[sp_map['SeqAggP'], :] >= 1.0"
    "    d_pct = float(np.mean(dead_arr) * 100.0)"
    "    pi_pct = float(np.mean(pi_d_arr) * 100.0)"
    "    p38_d_pct = float(np.mean(p38_d_arr) * 100.0)"
    "    inc_pct = float(np.mean(inc_arr) * 100.0)"
    "    lmax = 0.0"
    "    dt_lim = 0.0"
    "else:"
    "    pi_haz = max(float(y_ode[sp_map['PIdeath']]), 0.0)"
    "    p38_haz = max(float(y_ode[sp_map['p38death']]), 0.0)"
    "    d_pct = float((1.0 - np.exp(-(pi_haz + p38_haz))) * 100.0)"
    "    pi_pct = float((1.0 - np.exp(-pi_haz)) * 100.0)"
    "    p38_d_pct = float((1.0 - np.exp(-p38_haz)) * 100.0)"
    "    inc_pct = 100.0 if seq_v >= 1.0 else 0.0"
    "    p_now = scen.get_params_at(model, t_curr)"
    "    J = compute_jacobian(model, y_ode, p_now)"
    "    lmax = float(np.max(np.abs(np.linalg.eigvals(J))))"
    "    dt_lim = float(2.0 / lmax) if lmax > 0 else 999.0"
  )

  set sim-time-s py:runresult "float(t_curr)"
  set sim-time-h py:runresult "t_h"
  set curr-polyq-olig py:runresult "polyq_o"
  set curr-misp-olig py:runresult "misp_o"
  set curr-seq-agg py:runresult "seq_v"
  set curr-free-prot py:runresult "prot_f"
  set curr-seq-prot py:runresult "prot_s"
  set curr-ros py:runresult "ros_v"
  set curr-mrfpu py:runresult "mrfpu_v"
  set curr-p38-pct py:runresult "p38_pct"
  set curr-death-pct py:runresult "d_pct"
  set curr-pi-death-pct py:runresult "pi_pct"
  set curr-p38-death-pct py:runresult "p38_d_pct"
  set curr-inc-freq-pct py:runresult "inc_pct"
  set curr-lambda-max py:runresult "lmax"
  set curr-dt-max py:runresult "dt_lim"

  if t-seed-h < 0 and curr-seq-agg >= 1.0 [ set t-seed-h sim-time-h ]
  if t-exhaust-h < 0 and curr-free-prot < 1.0 [ set t-exhaust-h sim-time-h ]
end

to update-visuals
  ifelse solver-type = "Tau-Leap (Ensemble)" [
    let dead-list py:runresult "dead_arr.tolist()"
    let inc-list py:runresult "inc_arr.tolist()"
    let pi-list py:runresult "pi_d_arr.tolist()"
    ask cells [
      let d? item cell-id dead-list
      let inc? item cell-id inc-list
      let pi? item cell-id pi-list
      set is-dead? d?
      set has-inclusion? inc?
      ifelse d? [
        set color ifelse-value pi? [ magenta ] [ red ]
        set shape "x"
      ] [
        ifelse inc? [
          set color orange
          set shape "target"
        ] [
          set color sky
          set shape "circle"
        ]
      ]
    ]
  ] [
    let target-inc min list 80 round (curr-seq-agg / 250)
    let diff-inc (target-inc - count inclusions)
    if diff-inc > 0 [
      create-inclusions diff-inc [
        set shape "circle"
        set color orange
        set size 1.4
        move-to one-of patches with [ distancexy 0 0 < 8 ]
      ]
    ]
    if diff-inc < 0 [ ask n-of (- diff-inc) inclusions [ die ] ]

    let target-olig min list 100 round ((curr-polyq-olig + curr-misp-olig) / 30)
    let diff-olig (target-olig - count oligomers)
    if diff-olig > 0 [
      create-oligomers diff-olig [
        set shape "dot"
        set color yellow
        set size 0.9
        move-to one-of patches with [ distancexy 0 0 < 14 and distancexy 0 0 > 4 ]
      ]
    ]
    if diff-olig < 0 [ ask n-of (- diff-olig) oligomers [ die ] ]

    let target-prot min list 60 round (curr-free-prot / 10)
    let diff-prot (target-prot - count proteasomes)
    if diff-prot > 0 [
      create-proteasomes diff-prot [
        set shape "square"
        set color green
        set size 0.8
        move-to one-of patches with [ distancexy 0 0 < 14 ]
      ]
    ]
    if diff-prot < 0 [ ask n-of (- diff-prot) proteasomes [ die ] ]

    let target-ros min list 120 round (curr-ros / 4)
    let diff-ros (target-ros - count ros-particles)
    if diff-ros > 0 [
      create-ros-particles diff-ros [
        set shape "star"
        set color violet
        set size 0.7
        move-to one-of patches with [ distancexy 0 0 < 14 ]
      ]
    ]
    if diff-ros < 0 [ ask n-of (- diff-ros) ros-particles [ die ] ]

    ask oligomers [ rt random 60 - 30 fd 0.15 ]
    ask ros-particles [ rt random 90 - 45 fd 0.25 ]
    ask proteasomes [ rt random 40 - 20 fd 0.1 ]
  ]
end
@#$#@#$#@
GRAPHICS-WINDOW
355
10
777
433
-1
-1
12.5
1
10
1
1
1
0
1
1
1
-16
16
-16
16
1
1
1
ticks
30.0

BUTTON
10
10
95
45
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
105
10
195
45
Go / Pause
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
205
10
340
45
Step Once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

CHOOSER
10
55
340
100
scenario-mode
scenario-mode
"untreated" "p38_inhibitor" "pi_24h" "pi_24h_p38_inhibitor" "no_p38_ros_feedback" "pi_24h_no_feedback" "p38_inh_no_feedback" "no_basal_ros_gen" "no_polyq_control" "no_polyq_no_agg"
0

CHOOSER
10
105
185
150
solver-type
solver-type
"RK2 (ODE)" "Euler (ODE)" "Tau-Leap (Ensemble)"
0

SLIDER
190
105
340
150
step-size-dt
step-size-dt
0.5
30.0
2.0
0.5
1
s
HORIZONTAL

SLIDER
10
155
175
188
steps-per-tick
steps-per-tick
1
300
60
1
1
NIL
HORIZONTAL

SLIDER
180
155
340
188
n-ensemble-cells
n-ensemble-cells
25
300
300
25
1
cells
HORIZONTAL

SWITCH
10
195
175
228
mass-action-dimer?
mass-action-dimer?
0
1
-1000

SWITCH
180
195
340
228
use-custom-sliders?
use-custom-sliders?
1
1
-1000

SLIDER
10
235
175
268
kp38act-slider
kp38act-slider
0.0
1.0
1.0
0.05
1
NIL
HORIZONTAL

SLIDER
180
235
340
268
kproteff-24h-slider
kproteff-24h-slider
0.0
1.0
1.0
0.05
1
NIL
HORIZONTAL

SLIDER
10
273
175
306
kgenROSp38-slider
kgenROSp38-slider
0.0
0.002
0.0007
0.0001
1
NIL
HORIZONTAL

SLIDER
180
273
340
306
kaggMisP-slider
kaggMisP-slider
0.0
0.000001
0.00000005
0.00000001
1
NIL
HORIZONTAL

MONITOR
10
315
115
360
Time (hours)
precision sim-time-h 2
17
1
11

MONITOR
120
315
225
360
Total Death (%)
precision curr-death-pct 2
17
1
11

MONITOR
230
315
340
360
Inclusions (%)
precision curr-inc-freq-pct 1
17
1
11

MONITOR
10
365
115
410
Seed Time (h)
ifelse-value (t-seed-h >= 0) [ precision t-seed-h 2 ] [ "None" ]
17
1
11

MONITOR
120
365
225
410
Prot Exhaust (h)
ifelse-value (t-exhaust-h >= 0) [ precision t-exhaust-h 2 ] [ "None" ]
17
1
11

MONITOR
230
365
340
410
dt_max Limit (s)
ifelse-value (curr-dt-max > 0) [ precision curr-dt-max 2 ] [ "N/A (MC)" ]
17
1
11

MONITOR
10
415
340
460
Solver Status
status-msg
17
1
11

PLOT
785
10
1160
220
1. Oligomers & Inclusion Growth (Fig 1A)
Time (h)
Copies
0.0
48.0
0.0
2500.0
true
true
"" ""
PENS
"PolyQ Oligomers" 1.0 0 -955883 true "" "plotxy sim-time-h curr-polyq-olig"
"MisP Oligomers" 1.0 0 -2674135 true "" "plotxy sim-time-h curr-misp-olig"
"SeqAggP (x0.1)" 1.0 0 -13345367 true "" "plotxy sim-time-h (curr-seq-agg * 0.1)"

PLOT
1170
10
1545
220
2. Proteasome Depletion & mRFPu (Fig 1B & Fig 5A)
Time (h)
Copies
0.0
48.0
0.0
1500.0
true
true
"" ""
PENS
"Free Proteasome" 1.0 0 -10899396 true "" "plotxy sim-time-h curr-free-prot"
"Bound Proteasome" 1.0 0 -2674135 true "" "plotxy sim-time-h curr-seq-prot"
"mRFPu Reporter" 1.0 0 -13345367 true "" "plotxy sim-time-h curr-mrfpu"

PLOT
785
230
1160
440
3. ROS & p38 MAPK Feedback Loop (Fig 1C & Fig 5B)
Time (h)
Value
0.0
48.0
0.0
400.0
true
true
"" ""
PENS
"ROS Copies" 1.0 0 -8630108 true "" "plotxy sim-time-h curr-ros"
"Active p38_P (%)" 1.0 0 -7500403 true "" "plotxy sim-time-h curr-p38-pct"

PLOT
1170
230
1545
440
4. Cumulative Cell Death & Inclusion Freq (Fig 2, 3, 6)
Time (h)
Percent (%)
0.0
48.0
0.0
100.0
true
true
"" ""
PENS
"Total Death (%)" 1.0 0 -16777216 true "" "plotxy sim-time-h curr-death-pct"
"PI Route (%)" 1.0 0 -2064490 true "" "plotxy sim-time-h curr-pi-death-pct"
"p38 Route (%)" 1.0 0 -11221820 true "" "plotxy sim-time-h curr-p38-death-pct"
"Inclusion Freq (%)" 1.0 0 -955883 true "" "plotxy sim-time-h curr-inc-freq-pct"

PLOT
355
440
775
610
5. Explicit Stability Ceiling dt_max = 2/|lambda_max| (Fig 4B)
Time (h)
Seconds (s)
0.0
48.0
0.0
25.0
true
true
"" ""
PENS
"dt_max Ceiling (s)" 1.0 0 -2674135 true "" "if curr-dt-max > 0 [ plotxy sim-time-h curr-dt-max ]"
"Current Step dt (s)" 1.0 0 -13345367 true "" "plotxy sim-time-h step-size-dt"

@#$#@#$#@
## WHAT IS IT?
Interactive Tang et al. (2010) Huntington's PolyQ Model.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

circle
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

square
false
0
Rectangle -7500403 true true 30 30 270 270

star
false
0
Polygon -7500403 true true 150 0 185 105 300 105 210 175 240 285 150 215 60 285 90 175 0 105 115 105

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 false true 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 false true 90 90 120
Circle -7500403 true true 120 120 60

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
