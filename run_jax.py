import jax
from jax import random, config, jit, vmap
from graphfunc400 import graphfunc_jit
# from graphfunc600 import graphfunc_jit

config.update("jax_enable_x64", True)

key = random.PRNGKey(0)
N = 4
if N == 4:
    leafsize = 175
elif N == 6:
    leafsize = 1943

for batchsize in [10**i for i in range(0,8)]:
    %time x_jax = jax.device_put(random.uniform(key, shape=(batchsize, leafsize)))
    fvmap_jit = jit(vmap(graphfunc_jit))

    # %time fvmap_jit(x_jax).block_until_ready()
    # %timeit fvmap_jit(x_jax).block_until_ready()
    %time root = fvmap_jit(x_jax)
    for i in range(N):
        root[i].block_until_ready()

    %timeit root = fvmap_jit(x_jax)
    for i in range(N):
        root[i].block_until_ready()
