"""Tabulated initial fields shared by simulation builders."""
from .initial_conditions import load_initial_profile

def _unit_plane_noise(jax, jnp, key, shape, dtype):
    noise = jax.random.uniform(key, shape, dtype, minval=-0.5, maxval=0.5)
    noise -= jnp.mean(noise, axis=(-2, -1), keepdims=True)
    rms = jnp.sqrt(jnp.mean(noise * noise, axis=(-2, -1), keepdims=True))
    return noise / jnp.maximum(rms, jnp.finfo(dtype).tiny)


def initial_fields(case, jax, jnp):
    """Materialize the shared tabulated mean-plus-RMS initial state."""

    table = load_initial_profile(case)
    grid = case.physical_grid
    dtype = getattr(jnp, case.dtype)
    shape = (grid.nz, grid.ny, grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(case.initial_condition.seed), 3)
    u_noise = _unit_plane_noise(jax, jnp, keys[0], shape, dtype)
    v_noise = _unit_plane_noise(jax, jnp, keys[1], shape, dtype)
    coupled_noise = _unit_plane_noise(jax, jnp, keys[2], shape, dtype)

    def profile(name: str):
        return jnp.asarray(table[name], dtype)[:, None, None]

    u = profile("u_m_s") + profile("u_rms_m_s") * u_noise
    v = profile("v_m_s") + profile("v_rms_m_s") * v_noise
    w_upper = profile("w_upper_m_s") + profile("w_upper_rms_m_s") * coupled_noise
    w = jnp.concatenate((jnp.zeros_like(w_upper[:1]), w_upper), axis=0)
    w = w.at[-1].set(0.0)
    scalar = profile("scalar") + profile("scalar_rms") * coupled_noise
    return u, v, w, scalar
