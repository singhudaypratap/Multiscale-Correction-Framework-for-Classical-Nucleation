import numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from stockmayer_ewald_gpu import initial_state, make_kvectors, full_energy, trial_delta, MU_STAR
rng=np.random.default_rng(7)
n=20; rho=.8; pos,ori,L=initial_state(n,rho,7)
alpha=8/L; kv=jnp.asarray(make_kvectors(10)); E=0.4
U,S=full_energy(jnp.asarray(pos),jnp.asarray(ori),L,alpha,kv,E,rho)
idx=5; newp=(jnp.asarray(pos[idx])+jnp.array([.037,-.021,.013]))%L
axis=jnp.array([.2,.7,-.3]); axis=axis/jnp.linalg.norm(axis); angle=.17
u=jnp.asarray(ori[idx]); c=jnp.cos(angle); s=jnp.sin(angle); newu=u*c+jnp.cross(axis,u)*s+axis*jnp.dot(axis,u)*(1-c)
dU,dS=trial_delta(jnp.asarray(pos),jnp.asarray(ori),S,U,L,alpha,kv,idx,newp,newu,E)
newpos=jnp.asarray(pos).at[idx].set(newp); newori=jnp.asarray(ori).at[idx].set(newu)
U2,S2=full_energy(newpos,newori,L,alpha,kv,E,rho)
print('incremental/full delta:',float(dU),float(U2-U),'abs error',float(abs(dU-(U2-U))))
print('structure-factor update error:',float(jnp.max(jnp.abs((S+dS)-S2))))
assert float(abs(dU-(U2-U))) < 1e-7
assert float(jnp.max(jnp.abs((S+dS)-S2))) < 1e-10
print('ALL INCREMENTAL EWALD CHECKS PASSED')
