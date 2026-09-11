"""Final reduced-order CNT calculation for the manuscript. (v2 - bugfixed)

Scientific scope
----------------
This script is a thermodynamic screening model, not an atomistic nucleation
simulation. It uses IAPWS-95 liquid density and the IAPWS R1-76(2014) surface
tension correlation for 253-313 K. The 233 K case is intentionally rejected
as a primary water-property calculation because the IAPWS surface-tension
release only documents reasonable supercooled extrapolation to -25 C.

BUGFIX NOTES (v2, relative to the originally delivered version)
-----------------------------------------------------------------
1. CLI parsing: numeric-list arguments (T, E, S, delta_nm, c) are now parsed
   with nargs='+' (space-separated), not comma-joined strings. The original
   comma-string parsing broke argparse's negative-number handling: a token
   like "-0.05,0,0.05,0.10" does not match argparse's pure-negative-number
   regex (only "-0.05" alone would), so argparse treated it as an unknown
   option and the documented command line crashed with
   "error: argument --delta_nm: expected one argument".

2. Saturation pressure: the original code called IAPWS95(T=T, x=0) to get
   the saturation pressure. This packages's (T,x) solver hard-rejects any
   T below the water triple point (273.16 K) with
   "NotImplementedError: Incoming out of bound" -- independent of, and
   lower than, this script's own 248.15 K guard, so T=253 and T=273 K (two
   of the four documented default temperatures) crashed outright.
   Fixed by computing the saturation pressure from the IAPWS-95 auxiliary
   vapor-pressure correlation (Wagner & Pruss 2002) directly, which has a
   closed form and does not invoke the package's (T,x) solver. This
   matches the package's own (T,x) output to ~4 significant figures for
   T > Tt (verified at 293 K and 313 K), and gives the standard
   documented extrapolation below the triple point.

3. Units AND a conceptual pressure error: the original code computed
       P_MPa = sat.P * S / 1000.0
   but `sat.P` from the iapws package is already in MPa, so this divided
   the pressure by 1000 a second time. Beyond the raw unit bug, evaluating
   the LIQUID density at S times the (tiny) VAPOR saturation pressure is
   itself a physical mix-up: classical nucleation theory's driving-force
   term needs the bulk liquid density at its own reference state, not the
   vapor-side partial pressure -- liquid water is essentially incompressible
   over this whole pressure range (density changes by <0.1% between 0.001
   and 0.1 MPa), so the supersaturation ratio S has no meaningful effect on
   liquid density and should not be used to set the pressure at which the
   liquid EOS is evaluated. Compounding this, IAPWS95's (T,P)-mode phase
   solver in this package is numerically unreliable at the very low
   pressures (~1e-4 to 1e-3 MPa) that S*Psat produces at low T -- it
   sometimes returns the vapor-branch density even when the true
   equilibrium phase is liquid (verified directly: at T=253 K it flips
   from "Vapour" to "Liquid" somewhere between P=3.7e-4 and P=1e-3 MPa,
   a resolution artifact of the package's internal solver, not real
   physics). Fixed by evaluating the liquid EOS at a fixed, safe reference
   pressure (1 atm = 0.101325 MPa) for all T -- comfortably clear of the
   package's low-pressure artifact, and correct to <0.1% for this purpose
   regardless of S. IAPWS95 is called directly in (T, P) mode, which also
   avoids the (T,x) solver's separate hard triple-point restriction.

The electric-field term is written in a nonlinear permanent-dipole form:
    Δg_field = 1/2 α_e E^2 + kT ln[sinh(x)/x], x=p0 E/(kT)
so that its small-field expansion recovers the Debye quadratic term. This is
more defensible than using the quadratic expansion all the way to 1 GV/m.
The cooperative factor multiplies the per-molecule field free-energy gain and
is explicitly a phenomenological sensitivity parameter, not a calibrated
material constant.

The corrected critical radius is the physical stationary root of
    ΔGv_eff(r) (r+2δ)^2 = 2 σ∞ (r+3δ)
with N*(r)=4πr^3 n_l/3 evaluated self-consistently.
"""
from __future__ import annotations
import argparse
import math
import warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

try:
    from iapws.iapws95 import IAPWS95
except ImportError as exc:
    raise SystemExit("Install the real IAPWS package first: pip install -U iapws") from exc

K_B = 1.380649e-23
EPS0 = 8.8541878128e-12
R = 8.31446261815324
NA = 6.02214076e23
E_CHARGE = 1.602176634e-19
MW = 0.018015268
TC = 647.096
PC_MPA = 22.064
ALPHA_E_VOL = 1.45e-30
ALPHA_E_SI = 4.0 * math.pi * EPS0 * ALPHA_E_VOL
P0 = 1.85 * 3.33564e-30

# IAPWS-95 auxiliary saturation-pressure correlation (Wagner & Pruss 2002),
# closed form -- avoids invoking IAPWS95's own (T,x) solver, which hard-
# rejects T below the triple point (273.16 K) regardless of this script's
# own supercooled-extrapolation policy.
_WAGNER_A = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)
_WAGNER_E = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)


def psat_wagner(T: float) -> float:
    """Saturation vapor pressure of water (MPa), Wagner & Pruss (2002)
    auxiliary equation. Formally valid T_triple <= T <= Tc; used here with
    the same documented-extrapolation philosophy as sigma_iapws() for
    T >= 248.15 K. Verified to match iapws.IAPWS95's own (T,x=0) solver to
    ~4 significant figures for T > 273.16 K."""
    theta = 1.0 - T / TC
    lnp_over_pc = (TC / T) * sum(a * theta**e for a, e in zip(_WAGNER_A, _WAGNER_E))
    return PC_MPA * math.exp(lnp_over_pc)


def sigma_iapws(T: float) -> float:
    if T < 248.15:
        raise ValueError("IAPWS R1-76 surface-tension extrapolation is not documented below 248.15 K.")
    tau = 1.0 - T / TC
    return 235.8e-3 * tau**1.256 * (1.0 - 0.625 * tau)


P_REFERENCE_MPA = 0.101325

# Continuum-CNT validity classification.
# N>=1 remains the numerical stationary-root sanity condition; these
# thresholds are reporting/interpretation flags, not root-selection criteria.
CNT_VALIDITY_N_BELOW = 10.0
CNT_VALIDITY_N_CAUTION = 20.0
  # 1 atm; liquid density is evaluated here for all T (see FIX 3 above)


def rho_liquid_iapws95(T: float, S: float) -> float:
    if T < 248.15:
        raise ValueError("Do not use the primary IAPWS liquid-property calculation below 248.15 K.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # expected "extrapolated values" warning below Tt=273.16K
        state = IAPWS95(T=T, P=P_REFERENCE_MPA)  # FIX: direct (T,P) mode at a safe reference pressure
    if state.rho < 500.0:
        raise RuntimeError(
            f"IAPWS95 returned a vapor-branch density ({state.rho:.3e} kg/m^3) at "
            f"T={T} K, P={P_REFERENCE_MPA} MPa; expected liquid (~1000 kg/m^3)."
        )
    return state.rho


def nonlinear_field_gain_per_molecule(T: float, E: float) -> float:
    """Positive free-energy gain (J/molecule) from electronic + orientational response."""
    if E == 0.0:
        return 0.0
    x = P0 * E / (K_B * T)
    if abs(x) < 1e-4:
        log_sinh_over_x = x*x/6.0 - x**4/180.0 + x**6/2835.0
    else:
        log_sinh_over_x = math.log(math.sinh(x) / x)
    return 0.5 * ALPHA_E_SI * E**2 + K_B * T * log_sinh_over_x


def baseline_dgv(T: float, S: float, rho: float) -> float:
    return rho * R * T * math.log(S) / MW


def field_dgv(T: float, E: float, S: float, rho: float, N: float | None, c: float) -> float:
    gain = nonlinear_field_gain_per_molecule(T, E)
    if N is not None:
        gain *= 1.0 + c / N**(1.0 / 3.0)
    n_l = rho / MW * NA
    return baseline_dgv(T, S, rho) + n_l * gain


def standard_cnt(T: float, E: float, S: float = 5.0):
    rho = rho_liquid_iapws95(T, S)
    sigma = sigma_iapws(T)
    dgv = field_dgv(T, E, S, rho, None, 0.0)
    r = 2.0 * sigma / dgv
    G = 16.0 * math.pi * sigma**3 / (3.0 * dgv**2)
    n_l = rho / MW * NA
    N = 4.0 * math.pi * r**3 * n_l / 3.0
    return r, G, N, rho, dgv


def corrected_cnt(T: float, E: float, delta: float, S: float = 5.0, c: float = 0.15):
    rho = rho_liquid_iapws95(T, S)
    sigma_inf = sigma_iapws(T)
    n_l = rho / MW * NA

    def f(r):
        N = 4.0 * math.pi * r**3 * n_l / 3.0
        dgv = field_dgv(T, E, S, rho, N, c)
        return dgv * (r + 2.0 * delta)**2 - 2.0 * sigma_inf * (r + 3.0 * delta)

    # FIX 4 (root selection): as r -> 0, N -> 0 too, and the cooperative
    # term (1 + c/N^(1/3)) diverges, which manufactures a spurious sign
    # change in f(r) at an unphysical near-zero radius (N << 1 molecule).
    # That spurious root is a local MINIMUM of DeltaG(r), not the barrier.
    # Since f(r) = -(r+2*delta)^2/(4*pi*r^2) * d(DeltaG)/dr, the physical
    # barrier (a local MAXIMUM of DeltaG) is the root where f goes from
    # negative to positive as r increases -- not simply the first root
    # found. We also require N >= 1 at the root as a sanity floor.
    lo = max(1e-12, -2.0 * delta + 1e-12)
    hi = 10e-9
    grid = np.geomspace(lo, hi, 4000)
    vals = np.array([f(r) for r in grid])
    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0 and fa < 0 < fb:
            r_candidate = brentq(f, a, b, xtol=1e-15, rtol=1e-12)
            N_candidate = 4.0 * math.pi * r_candidate**3 * n_l / 3.0
            if N_candidate >= 1.0:
                roots.append(r_candidate)
    if not roots:
        raise RuntimeError(f"No physical (maximum, N>=1) stationary root: T={T}, E={E}, delta={delta}, S={S}, c={c}")

    r = roots[0]
    N = 4.0 * math.pi * r**3 * n_l / 3.0
    dgv = field_dgv(T, E, S, rho, N, c)
    sigma_r = sigma_inf / (1.0 + 2.0 * delta / r)
    G = 4.0 * math.pi * r**2 * sigma_r - (4.0 * math.pi / 3.0) * r**3 * dgv
    return r, G, N, rho, dgv


def generate(out: Path, temperatures, fields, supersaturations, deltas, c_values):
    rows = []
    n_skipped = 0
    for S in supersaturations:
        for T in temperatures:
            if T < 248.15:
                n_skipped += 1
                continue
            for E in fields:
                rs, Gs, Ns, rho, dgv = standard_cnt(T, E, S)
                for c in c_values:
                    for delta in deltas:
                        rc, Gc, Nc, _, _ = corrected_cnt(T, E, delta, S, c)
                        rows.append({
                            "T_K": T, "S": S, "E_Vpm": E, "delta_nm": delta*1e9,
                            "c": c, "rho_kg_m3": rho, "sigma_N_m": sigma_iapws(T),
                            "r_std_nm": rs*1e9, "N_std": Ns, "G_std_eV": Gs/E_CHARGE,
                            "r_corr_nm": rc*1e9, "N_corr": Nc, "G_corr_eV": Gc/E_CHARGE,
                            "barrier_change_pct": 100.0*(Gc-Gs)/Gs,
                            "field_gain_J_per_molecule": nonlinear_field_gain_per_molecule(T,E),
                        })
    if n_skipped:
        print(f"Skipped {n_skipped} temperature(s) below 248.15 K.")
    import csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corrected_CNT_final.csv")
    # FIX: nargs='+' (space-separated) instead of comma-joined strings, so
    # argparse's negative-number matcher correctly recognizes tokens like
    # "-0.05" as values rather than unknown options.
    ap.add_argument("--S", type=float, nargs="+", default=[3, 5, 10])
    ap.add_argument("--T", type=float, nargs="+", default=[253, 273, 293, 313])
    ap.add_argument("--E", type=float, nargs="+", default=[0, 1e8, 2e8, 4e8, 6e8, 8e8, 1e9])
    ap.add_argument("--delta_nm", type=float, nargs="+", default=[-0.05, 0, 0.05, 0.10])
    ap.add_argument("--c", type=float, nargs="+", default=[0, 0.075, 0.15, 0.30])
    args = ap.parse_args()
    d = [x * 1e-9 for x in args.delta_nm]
    rows = generate(Path(args.out), args.T, args.E, args.S, d, args.c)
    print(f"Wrote {len(rows)} corrected CNT rows to {args.out}")
