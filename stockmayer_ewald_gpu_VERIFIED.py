"""GPU Stockmayer NVT Monte Carlo with 3-D dipolar Ewald electrostatics.

This is a BULK STOCKMAYER-FLUID CONSISTENCY calculation, not a water
nucleation free-energy calculation. It is the production-quality replacement
for the earlier minimum-image Stockmayer code.

Model and units
---------------
Reduced LJ units are used: sigma = epsilon = k_B = 1.  The SPC/E Lennard-Jones
mapping is sigma=3.166 Angstrom, epsilon/k_B=78.19743111 K, and molecular dipole
2.35 D. The physical electric field is converted as E*=E*mu_scale/epsilon,
where mu_scale=sqrt(4*pi*eps0*epsilon*sigma^3).

Electrostatics
--------------
Point-dipole interactions use the standard 3-D Ewald decomposition with
conducting/tinfoil boundary conditions: real-space term, reciprocal-space
term, and dipole self term. The reciprocal structure factor is maintained
incrementally, so a trial move updates the reciprocal energy in O(N_k) rather
than recomputing the full structure factor. The real-space trial energy is
updated from the moved particle's pair interactions. This is the appropriate
kind of incremental Ewald bookkeeping for Metropolis MC.

LJ treatment
------------
The LJ interaction is truncated at r_c=L/2. A standard homogeneous long-range
dispersion energy correction is added to reported energies. The correction is
constant at fixed N,V, so it cancels exactly from Metropolis acceptance.

Statistics
----------
Independent replicas are run in parallel on the GPU with JAX. Each replica
remains a sequential Markov chain; replicas are the independent statistical
units. The script reports per-replica mean energy, temporal standard deviation,
mean z-orientation and acceptance.
"""
from __future__ import annotations
import argparse, csv, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.special import erfc

KB = 1.380649e-23
EPS0 = 8.8541878128e-12
SIGMA_M = 3.166e-10
EPS_K = 78.19743111
EPS_SI = KB * EPS_K
MU_D = 2.35
DEBYE = 3.33564e-30
MU_SI = MU_D * DEBYE
MU_SCALE = math.sqrt(4.0 * math.pi * EPS0 * EPS_SI * SIGMA_M**3)
MU_STAR = MU_SI / MU_SCALE
RHO_STAR = 0.80

@dataclass(frozen=True)
class EwaldParams:
    alpha_over_L: float = 8.0
    kmax: int = 10


def make_kvectors(kmax: int) -> np.ndarray:
    vals = np.arange(-kmax, kmax + 1, dtype=np.int32)
    gx, gy, gz = np.meshgrid(vals, vals, vals, indexing="ij")
    k = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return k[np.any(k != 0, axis=1)].astype(np.float64)


def initial_state(n: int, rho: float, seed: int):
    rng = np.random.default_rng(seed)
    L = (n / rho) ** (1.0 / 3.0)
    m = int(np.ceil(n ** (1.0 / 3.0)))
    a = L / m
    xyz = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                if len(xyz) < n:
                    xyz.append(((ix + 0.5) * a,
                                (iy + 0.5) * a,
                                (iz + 0.5) * a))
    pos = np.asarray(xyz, dtype=np.float64) % L
    # Tiny random displacement breaks the artificial lattice symmetry without
    # changing the prescribed density.
    pos = (pos + 0.02 * rng.normal(size=pos.shape)) % L
    ori = rng.normal(size=(n, 3))
    ori /= np.linalg.norm(ori, axis=1, keepdims=True)
    return pos, ori, L


def real_pair(u_i, u_j, rij, alpha):
    r2 = jnp.dot(rij, rij)
    r = jnp.sqrt(jnp.maximum(r2, 1e-24))
    invr2 = 1.0 / r2
    invr3 = invr2 / r
    invr4 = invr2 * invr2
    invr5 = invr4 / r
    er = erfc(alpha * r)
    ex = jnp.exp(-(alpha * r) ** 2)
    B = er * invr3 + (2.0 * alpha / jnp.sqrt(jnp.pi)) * ex * invr2
    C = 3.0 * er * invr5 + (2.0 * alpha / jnp.sqrt(jnp.pi)) * ex * (3.0 * invr4 + 2.0 * alpha**2 * invr2)
    return MU_STAR**2 * (B * jnp.dot(u_i, u_j) - C * jnp.dot(u_i, rij) * jnp.dot(u_j, rij))


def lj_pair(r2):
    inv2 = 1.0 / jnp.maximum(r2, 1e-24)
    inv6 = inv2**3
    return 4.0 * (inv6**2 - inv6)


def ewald_recip_structure(pos, ori, L, kvec_int):
    kv = (2.0 * jnp.pi / L) * kvec_int
    phase = pos @ kv.T
    kr = ori @ kv.T
    return jnp.sum(kr * jnp.exp(1j * phase), axis=0)


def full_energy(pos, ori, L, alpha, kvec_int, e_star, rho_star):
    n = pos.shape[0]
    dr = pos[:, None, :] - pos[None, :, :]
    dr = dr - L * jnp.round(dr / L)
    r2 = jnp.sum(dr * dr, axis=-1)
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), 1)
    rc2 = (0.5 * L) ** 2
    valid = (r2 > 1e-20) & (r2 < rc2) & (mask > 0)
    r = jnp.sqrt(jnp.maximum(r2, 1e-24))
    inv2 = 1.0 / jnp.maximum(r2, 1e-24)
    inv6 = inv2**3
    u_lj = jnp.sum(jnp.where(valid, 4.0 * (inv6**2 - inv6), 0.0))

    dotmu = ori @ ori.T
    mui_r = jnp.einsum("ik,ijk->ij", ori, dr)
    muj_r = jnp.einsum("jk,ijk->ij", ori, dr)
    er = erfc(alpha * r)
    ex = jnp.exp(-(alpha * r)**2)
    B = er / (r**3) + (2*alpha/jnp.sqrt(jnp.pi))*ex/(r**2)
    C = 3*er/(r**5) + (2*alpha/jnp.sqrt(jnp.pi))*ex*(3/(r**4)+2*alpha**2/(r**2))
    u_real = MU_STAR**2 * jnp.sum(jnp.where(valid, B*dotmu - C*mui_r*muj_r, 0.0))

    kv = (2*jnp.pi/L)*kvec_int
    k2 = jnp.sum(kv*kv, axis=1)
    S = ewald_recip_structure(pos, ori, L, kvec_int)
    coeff = jnp.exp(-k2/(4*alpha**2))/k2
    u_recip = MU_STAR**2 * (2*jnp.pi/L**3) * jnp.sum(coeff*jnp.real(S*jnp.conj(S)))
    u_self = -(2*alpha**3/(3*jnp.sqrt(jnp.pi))) * n * MU_STAR**2

    rc = 0.5*L
    u_tail_particle = (8*jnp.pi*rho_star/3.0) * ((1.0/3.0)*(1.0/rc**9) - 1.0/(rc**3))
    u_lj = u_lj + n*u_tail_particle
    u_field = -e_star*MU_STAR*jnp.sum(ori[:,2])
    return u_lj + u_real + u_recip + u_self + u_field, S


def moved_real_delta(pos, ori, idx, new_pos, new_ori, L, alpha):
    old_p = pos[idx]; old_u = ori[idx]
    dr_old = old_p - pos
    dr_old = dr_old - L*jnp.round(dr_old/L)
    dr_new = new_pos - pos
    dr_new = dr_new - L*jnp.round(dr_new/L)
    mask = jnp.arange(pos.shape[0]) != idx
    rc2 = (0.5*L)**2
    old_r2 = jnp.sum(dr_old*dr_old, axis=1)
    new_r2 = jnp.sum(dr_new*dr_new, axis=1)
    old_valid = mask & (old_r2 < rc2) & (old_r2 > 1e-20)
    new_valid = mask & (new_r2 < rc2) & (new_r2 > 1e-20)
    old_lj = jnp.where(old_valid, lj_pair(old_r2), 0.0)
    new_lj = jnp.where(new_valid, lj_pair(new_r2), 0.0)
    old_d = vmap(real_pair, in_axes=(None,0,0,None))(old_u, ori, dr_old, alpha)
    new_d = vmap(real_pair, in_axes=(None,0,0,None))(new_ori, ori, dr_new, alpha)
    old_d = jnp.where(old_valid, old_d, 0.0)
    new_d = jnp.where(new_valid, new_d, 0.0)
    return jnp.sum(new_lj + new_d) - jnp.sum(old_lj + old_d)


def trial_delta(pos, ori, S, total_u, L, alpha, kvec_int, idx, new_pos, new_ori, e_star):
    dr_real = moved_real_delta(pos, ori, idx, new_pos, new_ori, L, alpha)
    kv = (2*jnp.pi/L)*kvec_int
    old_term = (ori[idx] @ kv.T) * jnp.exp(1j*(pos[idx] @ kv.T))
    new_term = (new_ori @ kv.T) * jnp.exp(1j*(new_pos @ kv.T))
    dS = new_term - old_term
    k2 = jnp.sum(kv*kv, axis=1)
    coeff = MU_STAR**2 * (2*jnp.pi/L**3) * jnp.exp(-k2/(4*alpha**2))/k2
    drec = jnp.sum(coeff * (2*jnp.real(jnp.conj(S)*dS) + jnp.real(dS*jnp.conj(dS))))
    dfield = -e_star*MU_STAR*(new_ori[2]-ori[idx,2])
    return dr_real + drec + dfield, dS


def one_replica(key, pos0, ori0, L, Tstar, e_star, alpha, kvec_int,
                rho_star, n_eq, n_prod, max_disp, max_angle, sample_stride):
    n = pos0.shape[0]
    beta = 1.0/Tstar
    total_u, S0 = full_energy(pos0, ori0, L, alpha, kvec_int, e_star, rho_star)

    def move(step, carry):
        key, pos, ori, S, U, accepted = carry
        key, ikey, dkey, akey, okey, angkey, rkey = jax.random.split(key, 7)
        idx = jax.random.randint(ikey, (), 0, n)
        translate = jax.random.uniform(dkey) < 0.5
        disp = (jax.random.uniform(akey, (3,))*2-1)*max_disp
        axis = jax.random.normal(okey, (3,)); axis /= jnp.linalg.norm(axis)
        angle = (jax.random.uniform(angkey)*2-1)*max_angle
        oldp, oldu = pos[idx], ori[idx]
        newp = jax.lax.cond(translate, lambda _: (oldp+disp)%L, lambda _: oldp, operand=None)
        newu = jax.lax.cond(translate, lambda _: oldu,
                             lambda _: rotate(oldu, axis, angle), operand=None)
        du, dS = trial_delta(pos, ori, S, U, L, alpha, kvec_int, idx, newp, newu, e_star)
        accept = (du <= 0) | (jax.random.uniform(rkey) < jnp.exp(-beta*jnp.minimum(du, 700.0)))
        pos = lax.cond(accept, lambda _: pos.at[idx].set(newp), lambda _: pos, operand=None)
        ori = lax.cond(accept, lambda _: ori.at[idx].set(newu), lambda _: ori, operand=None)
        S = lax.cond(accept, lambda _: S+dS, lambda _: S, operand=None)
        U = U + jnp.where(accept, du, 0.0)
        return key, pos, ori, S, U, accepted+accept.astype(jnp.int32)

    key, pos, ori, S, U, acc_eq = lax.fori_loop(0, n_eq*n, move,
                                                  (key,pos0,ori0,S0,total_u,jnp.int32(0)))
    init = (key,pos,ori,S,U,jnp.int32(0),jnp.float64(0),jnp.float64(0),jnp.float64(0),jnp.int32(0))
    def prod(step, carry):
        key,pos,ori,S,U,acc,se,se2,sm,ns = carry
        key,pos,ori,S,U,a = move(step,(key,pos,ori,S,U,jnp.int32(0)))
        take = (step % sample_stride)==0
        se = se + jnp.where(take,U,0.0)
        se2 = se2 + jnp.where(take,U*U,0.0)
        sm = sm + jnp.where(take,jnp.mean(ori[:,2]),0.0)
        ns = ns + take.astype(jnp.int32)
        acc = acc + a
        return key,pos,ori,S,U,acc,se,se2,sm,ns
    key,pos,ori,S,U,acc,se,se2,sm,ns = lax.fori_loop(0,n_prod*n,prod,init)
    mean = se/ns; sd=jnp.sqrt(jnp.maximum(se2/ns-mean**2,0.0))
    return mean,sd,sm/ns,acc/(n_prod*n)


def rotate(u, axis, angle):
    c=jnp.cos(angle); s=jnp.sin(angle); d=jnp.dot(axis,u)
    return u*c+jnp.cross(axis,u)*s+axis*d*(1-c)


def run_case(n,T_K,E_Vpm,replicas=3,seed=20260910,rho_star=RHO_STAR,
             n_eq=3000,n_prod=5000,max_disp=0.08,max_angle=0.20,sample_stride=10,
             ewald=EwaldParams()):
    L=(n/rho_star)**(1/3); alpha=ewald.alpha_over_L/L
    Tstar=T_K/EPS_K; e_star=E_Vpm*MU_SCALE/EPS_SI
    kv=jnp.asarray(make_kvectors(ewald.kmax))
    pos=[];ori=[]
    for r in range(replicas):
        p,o,_=initial_state(n,rho_star,seed+r+10000);pos.append(p);ori.append(o)
    keys=jax.random.split(jax.random.PRNGKey(seed),replicas)
    fn=jax.jit(vmap(one_replica,in_axes=(0,0,0,None,None,None,None,None,None,None,None,None,None,None),out_axes=0))
    vals=fn(keys,jnp.asarray(np.stack(pos)),jnp.asarray(np.stack(ori)),L,Tstar,e_star,alpha,kv,
            rho_star,n_eq,n_prod,max_disp,max_angle,sample_stride)
    vals=tuple(np.asarray(x) for x in jax.device_get(vals))
    return vals,dict(N=n,T_K=T_K,E_Vpm=E_Vpm,Tstar=Tstar,Estar=e_star,rho_star=rho_star,L_star=L,alpha_ewald=alpha)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='stockmayer_ewald_gpu_results.csv')
    ap.add_argument('--replicas',type=int,default=3)
    ap.add_argument('--neq',type=int,default=3000)
    ap.add_argument('--nprod',type=int,default=5000)
    ap.add_argument('--sizes',default='20,50,100')
    ap.add_argument('--temps',default='273,293')
    ap.add_argument('--fields',default='0,5e8,1e9')
    args=ap.parse_args()
    rows=[]
    print('JAX devices:',jax.devices()); print(f'mu*={MU_STAR:.8f}; mu_scale={MU_SCALE:.6e} C m; epsilon={EPS_SI:.6e} J')
    for T in map(float,args.temps.split(',')):
      for n in map(int,args.sizes.split(',')):
       for E in map(float,args.fields.split(',')):
        print(f'Running N={n}, T={T:g} K, E={E:.3e} V/m')
        vals,meta=run_case(n,T,E,args.replicas,n_eq=args.neq,n_prod=args.nprod)
        for r in range(args.replicas):
          rows.append(dict(replica=r,**meta,
                           mean_Estar=vals[0][r],sd_Estar_time=vals[1][r],mean_mz=vals[2][r],acceptance=vals[3][r]))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('Wrote',out)

if __name__=='__main__':main()
