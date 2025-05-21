import numpy as np
import plotly.graph_objects as go

def lagrangian_particle_model(total_particles,
                              time_steps,
                              wind_speed,
                              turbulence_intensity,
                              dt,
                              enable_3d,
                              st,
                              tau,
                              sources,
                              p_absorb=0.1,           # ⬅ 新增：吸收概率 p
                              kx=0.8, ky=0.8, kz=0.9   # ⬅ 新增：速度倍率
                              ):
    """
    同名参数含义不变。
    新增：
      p_absorb :  0–1，粒子撞地被吸收的概率
      kx, ky, kz: 未吸收粒子在 x, y, z 方向速度的衰减倍率
    """

    dim = 3 if enable_3d else 2
    particles = np.zeros((total_particles, dim))
    vx_turb  = np.zeros(total_particles)
    vy_turb  = np.zeros(total_particles)
    vz_turb  = np.zeros(total_particles) if enable_3d else None
    source_labels = np.zeros(total_particles, dtype=int)


    idx = 0
    for s_id, (sx, sy, sz, n) in enumerate(sources):
        for _ in range(n):
            particles[idx, 0] = sx
            particles[idx, 1] = sy
            if enable_3d:
                particles[idx, 2] = sz
            source_labels[idx] = s_id
            idx += 1

    active = np.ones(total_particles, dtype=bool)   # 活跃粒子掩码


    for _ in range(time_steps):

        rand = np.random.randn
        vx_turb = vx_turb*(1-dt/tau) + turbulence_intensity*rand(total_particles)*np.sqrt(2*dt/tau)
        vy_turb = vy_turb*(1-dt/tau) + turbulence_intensity*rand(total_particles)*np.sqrt(2*dt/tau)
        if enable_3d:
            vz_turb = vz_turb*(1-dt/tau) + turbulence_intensity*rand(total_particles)*np.sqrt(2*dt/tau)

        particles[active, 0] += (wind_speed + vx_turb[active]) * dt
        particles[active, 1] += vy_turb[active] * dt
        if enable_3d:
            particles[active, 2] += vz_turb[active] * dt

            hit = active & (particles[:, 2] <= 0.0)
            if hit.any():
                rnd = np.random.rand(hit.sum())
                absorb = hit.copy()
                absorb[hit] = rnd < p_absorb       # true → 被吸收
                reflect = hit & (~absorb)          # true → 反弹

                particles[absorb] = np.nan
                vx_turb[absorb] = vy_turb[absorb] = vz_turb[absorb] = np.nan
                active[absorb] = False

                particles[reflect, 2] = 0.0        # 粒子贴地
                vx_turb[reflect] *= kx
                vy_turb[reflect] *= ky
                vz_turb[reflect] = -vz_turb[reflect] * kz


    fig = go.Figure()
    for s_id, _ in enumerate(sources):
        sel = (source_labels == s_id) & active     # 只画活跃粒子
        if sel.sum() == 0:
            continue

        if enable_3d:
            fig.add_trace(go.Scatter3d(
                x=particles[sel, 0],
                y=particles[sel, 1],
                z=particles[sel, 2],
                mode="markers",
                name=f"Source {s_id+1}",
                marker=dict(size=3)))
        else:
            fig.add_trace(go.Scatter(
                x=particles[sel, 0],
                y=particles[sel, 1],
                mode="markers",
                name=f"Source {s_id+1}",
                marker=dict(size=5)))


    fig.update_layout(title="Lagrangian Particle Tracking (with ground effects)")
    st.plotly_chart(fig)
