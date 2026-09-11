"""Numerical checks for the GPU Stockmayer Ewald implementation."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from stockmayer_ewald_gpu import full_energy, make_kvectors, MU_STAR

rng=np.random.default_rng(1234)
n=20; rho=.8; L=(n/rho)**(1/3)
m=int(np.ceil(n**(1/3))); a=L/m; pos=[]
for ix in range(m):
 for iy in range(m):
  for iz in range(m):
   if len(pos)<n: pos.append([(ix+.5)*a,(iy+.5)*a,(iz+.5)*a])
pos=(np.asarray(pos)+.03*rng.normal(size=(n,3)))%L
ori=rng.normal(size=(n,3)); ori/=np.linalg.norm(ori,axis=1,keepdims=True)

def E(pos,ori,alpha_over_L,kmax,field=0.0):
    return float(full_energy(jnp.asarray(pos),jnp.asarray(ori),L,alpha_over_L/L,jnp.asarray(make_kvectors(kmax)),field,rho)[0])

e1=E(pos,ori,9.0,14)
e2=E(pos,ori,8.0,12)
print('Ewald split stability |E(9,14)-E(8,12)|:',abs(e1-e2))
assert abs(e1-e2) < 1e-3
# Energy must be finite and field reversal must reverse only the external-field contribution.
e0=E(pos,ori,8.0,12,0.0)
eplus=E(pos,ori,8.0,12,0.4)
ominus=E(pos,ori,8.0,12,-0.4)
assert np.isfinite([e0,eplus,ominus]).all()
assert abs((eplus+ominus)-2*e0) < 1e-8
print('field antisymmetry check:',abs((eplus+ominus)-2*e0))
print('ALL EWALD CHECKS PASSED')
