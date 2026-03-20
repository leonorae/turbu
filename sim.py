"""
Active nematic turbulence simulation — Lavi et al. PRR 2026.
Equations (6)–(8) with pseudospectral solver.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


# ── Fourier helpers ────────────────────────────────────────────────────────────

def make_grid(N):
    k1d = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
    KX, KY = np.meshgrid(k1d, k1d, indexing='ij')
    K2 = KX**2 + KY**2
    K4 = K2**2
    K4_inv = np.where(K4 > 0, 1.0 / K4, 0.0)
    # 2/3 dealiasing mask
    kmax = N * np.pi * 2.0 / 3.0
    mask = (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)
    return KX, KY, K2, K4, K4_inv, mask


def d1_op(f_hat, KX, KY):
    """d1 = ½(∂y²−∂x²);  Fourier: ½(kx²−ky²)·f̂"""
    return 0.5 * (KX**2 - KY**2) * f_hat


def d2_op(f_hat, KX, KY):
    """d2 = ∂x∂y;  Fourier: −kx·ky·f̂"""
    return -KX * KY * f_hat


def rfft(f, mask):
    return np.fft.fft2(f) * mask


def irfft(fh):
    return np.real(np.fft.ifft2(fh))


# ── Solver ─────────────────────────────────────────────────────────────────────

class ActiveNematic:
    """
    Director-based defect-free active nematic turbulence.

    Non-dimensionalisation: length by L, time by τa = η/|ζ|.
    Parameters:
        N   : grid points per side
        A   : activity number = L² / (2 ℓa²)
        R   : viscosity ratio γ/η
        nu  : flow-alignment parameter (ν ≤ 0 for rods)
        S   : sign of activity (+1 extensile, −1 contractile)
        dt  : time step
        D   : noise amplitude
    """

    def __init__(self, N=64, A=3.2e5, R=1.0, nu=0.0, S=1, dt=0.02, D=2.5e-3):
        self.N = N
        self.A = A
        self.R = R
        self.nu = nu
        self.S = S
        self.dt = dt
        self.D = D
        self.t = 0.0

        self.KX, self.KY, self.K2, self.K4, self.K4_inv, self.mask = make_grid(N)

        # Initialise with small-amplitude noise
        rng = np.random.default_rng(42)
        self.theta = 0.05 * rng.standard_normal((N, N))
        self.psi   = np.zeros((N, N))

        # Previous-step explicit RHS for AB2
        self._prev_rhs = None

    # ── stream-function ────────────────────────────────────────────────────────

    def _solve_psi(self, theta):
        """
        Solve ∇⁴ψ = RHS(θ) − G(θ,ψ)  by fixed-point iteration.
        From Eq. (6) and Appendix A, Eq. (A1).
        """
        A, R, nu, S = self.A, self.R, self.nu, self.S
        KX, KY, K2, K4, K4_inv, mask = (
            self.KX, self.KY, self.K2, self.K4, self.K4_inv, self.mask)

        sin2t = np.sin(2 * theta)
        cos2t = np.cos(2 * theta)

        sin2t_h = rfft(sin2t, mask)
        cos2t_h = rfft(cos2t, mask)

        # --- Eq (A1) RHS (terms independent of ψ) ---

        # 1. Active: S(d1 sin2θ + d2 cos2θ)
        active_h = S * (d1_op(sin2t_h, KX, KY) + d2_op(cos2t_h, KX, KY))

        # 2. Elastic + Ericksen: −(R/A)[½∇⁴θ + (∂x∇²θ)∂yθ − (∂y∇²θ)∂xθ]
        th_h = np.fft.fft2(theta) * mask
        lap2_th_h = -K2 * th_h
        lap4_th_h =  K4 * th_h

        dx_lap2_th = irfft(1j * KX * lap2_th_h)
        dy_lap2_th = irfft(1j * KY * lap2_th_h)
        dx_th      = irfft(1j * KX * th_h)
        dy_th      = irfft(1j * KY * th_h)

        nonlin_h = rfft(dx_lap2_th * dy_th - dy_lap2_th * dx_th, mask)
        elastic_h = -(R / A) * (0.5 * lap4_th_h + nonlin_h)

        # 3. Flow-alignment with h⊥: +(Rν/A)[−d1(cos2θ ∇²θ) + d2(sin2θ ∇²θ)]
        #    (derived from Rν ∂α∂β(q̂αβ h⊥) in Eq. 6, with h⊥ = (1/A)∇²θ)
        if nu != 0.0:
            lap2_th = irfft(lap2_th_h)
            cos2t_lap2t_h = rfft(cos2t * lap2_th, mask)
            sin2t_lap2t_h = rfft(sin2t * lap2_th, mask)
            nu_h = (R * nu / A) * (
                -d1_op(cos2t_lap2t_h, KX, KY)
                +d2_op(sin2t_lap2t_h, KX, KY))
        else:
            nu_h = 0.0

        rhs_h = active_h + elastic_h + nu_h

        # --- fixed-point iteration (Appendix A §1) ---
        psi_h = np.fft.fft2(self.psi) * mask   # initial guess = ψ^{n−1}

        if nu == 0.0:
            # No G term → direct inversion
            psi_h = K4_inv * rhs_h
            psi_h[0, 0] = 0.0
        else:
            tol = 1e-8
            for _ in range(30):
                # h‖ = ν(sin2θ d1ψ + cos2θ d2ψ)
                d1_psi_h = d1_op(psi_h, KX, KY)
                d2_psi_h = d2_op(psi_h, KX, KY)
                d1_psi   = irfft(d1_psi_h)
                d2_psi   = irfft(d2_psi_h)
                h_par    = nu * (sin2t * d1_psi + cos2t * d2_psi)
                h_par_h  = rfft(h_par, mask)

                # G = Rν[d1(sin2θ h‖) + d2(cos2θ h‖)]
                G_h = R * nu * (
                    d1_op(rfft(sin2t * irfft(h_par_h), mask), KX, KY)
                    + d2_op(rfft(cos2t * irfft(h_par_h), mask), KX, KY))

                psi_h_new = K4_inv * (rhs_h - G_h)
                psi_h_new[0, 0] = 0.0

                diff = np.sum(np.abs(psi_h_new - psi_h)**2)
                norm = np.sum(np.abs(psi_h_new)**2) + 1e-30
                psi_h = psi_h_new
                if diff / norm < tol:
                    break

        psi_h[0, 0] = 0.0
        return irfft(psi_h), psi_h

    # ── time step ──────────────────────────────────────────────────────────────

    def step(self, n_sub=1):
        for _ in range(n_sub):
            self._step_once()

    def _step_once(self):
        A, R, nu, dt = self.A, self.R, self.nu, self.dt
        KX, KY, K2, mask = self.KX, self.KY, self.K2, self.mask

        theta = self.theta
        psi, psi_h = self._solve_psi(theta)
        self.psi = psi

        # Velocity and vorticity
        vx    = irfft( 1j * KY * psi_h)           # vx = ∂y ψ
        vy    = irfft(-1j * KX * psi_h)            # vy = −∂x ψ
        omega = irfft(K2 * psi_h)                  # ω = −∇²ψ  →  +K²ψ in Fourier

        # ∂xθ, ∂yθ
        th_h   = np.fft.fft2(theta) * mask
        dx_th  = irfft(1j * KX * th_h)
        dy_th  = irfft(1j * KY * th_h)

        # Advection: −(v·∇)θ
        advection = -(vx * dx_th + vy * dy_th)

        # Flow-alignment: −νC,  C = cos2θ d1ψ − sin2θ d2ψ
        if nu != 0.0:
            cos2t = np.cos(2 * theta)
            sin2t = np.sin(2 * theta)
            d1_psi = irfft(d1_op(psi_h, KX, KY))
            d2_psi = irfft(d2_op(psi_h, KX, KY))
            C = cos2t * d1_psi - sin2t * d2_psi
            fa = -nu * C
        else:
            fa = 0.0

        # Noise
        if self.D > 0.0:
            noise_h = np.fft.fft2(np.random.randn(self.N, self.N)) * mask
            noise_h[0, 0] = 0.0
            noise_scale = np.sqrt(3.0 * self.D / (dt * (1.0/self.N)**2))
            noise = noise_scale * irfft(noise_h)
        else:
            noise = 0.0

        explicit_rhs = advection + 0.5 * omega + fa + noise

        # Adams-Bashforth 2 (1st step: forward Euler)
        if self._prev_rhs is None:
            rhs_ab = explicit_rhs
        else:
            rhs_ab = 1.5 * explicit_rhs - 0.5 * self._prev_rhs
        self._prev_rhs = explicit_rhs

        # Semi-implicit update (implicit diffusion):
        #   (1 + dt K²/A) θ̂^{n+1} = θ̂^n + dt · RHS_explicit
        rhs_h = th_h + dt * np.fft.fft2(rhs_ab)
        th_new_h = rhs_h / (1.0 + dt * K2 / A)
        th_new_h[0, 0] = 0.0           # zero mean

        self.theta = irfft(th_new_h)
        self.t += dt

    # ── diagnostics ────────────────────────────────────────────────────────────

    def velocity(self):
        psi_h = np.fft.fft2(self.psi) * self.mask
        vx = irfft( 1j * self.KY * psi_h)
        vy = irfft(-1j * self.KX * psi_h)
        return vx, vy

    def frank_density(self):
        th_h  = np.fft.fft2(self.theta) * self.mask
        dx_th = irfft(1j * self.KX * th_h)
        dy_th = irfft(1j * self.KY * th_h)
        return dx_th**2 + dy_th**2

    def director(self):
        """Unit director field (nx, ny) = (cos θ, sin θ)"""
        return np.cos(self.theta), np.sin(self.theta)

    def vorticity(self):
        psi_h = np.fft.fft2(self.psi) * self.mask
        return irfft(self.K2 * psi_h)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.theta = 0.05 * rng.standard_normal((self.N, self.N))
        self.psi   = np.zeros((self.N, self.N))
        self._prev_rhs = None
        self.t = 0.0


# ── Interactive visualisation ──────────────────────────────────────────────────

PRESETS = {
    'Strong turbulence\n(S=−1, ν=−1.1)':   dict(S=-1, nu=-1.1),
    'Arrested turbulence\n(S=+1, ν=−1.1)': dict(S=+1, nu=-1.1),
    'Non-aligning\n(ν=0)':                  dict(S=+1, nu=0.0),
}

VIEW_MODES = ['Speed', 'Frank energy', 'Director angle', 'Vorticity', 'Stream fn']

STEPS_PER_FRAME = 4


def build_ui(N=64, A=3.2e5):
    sim = ActiveNematic(N=N, A=A, nu=0.0, S=1, dt=0.02, D=2.5e-3)
    running = [True]
    view    = [VIEW_MODES[1]]    # default: Frank energy
    frame_n = [0]

    # ── layout ──
    fig = plt.figure(figsize=(13, 8), facecolor='#1a1a2e')
    fig.suptitle('Active Nematic Turbulence  —  Lavi et al. PRR 2026',
                 color='white', fontsize=12, y=0.98)

    # Main axes: field + streamlines
    ax_main = fig.add_axes([0.04, 0.28, 0.52, 0.66])
    ax_main.set_aspect('equal')
    ax_main.axis('off')

    # Spectra axes (Frank energy vs q)
    ax_spec = fig.add_axes([0.60, 0.55, 0.37, 0.38])
    ax_spec.set_facecolor('#0d0d1a')
    ax_spec.tick_params(colors='white', labelsize=8)
    for sp in ax_spec.spines.values():
        sp.set_color('#444')
    ax_spec.set_xlabel('|q| / (2π)', color='white', fontsize=8)
    ax_spec.set_ylabel('Frank energy spectrum', color='white', fontsize=8)
    ax_spec.set_title('Fourier spectra', color='#aaa', fontsize=9)

    # Info text
    ax_info = fig.add_axes([0.60, 0.42, 0.37, 0.11])
    ax_info.axis('off')
    info_txt = ax_info.text(0.0, 0.95, '', color='white', fontsize=8.5,
                            va='top', fontfamily='monospace',
                            transform=ax_info.transAxes)

    # ── sliders ──
    sl_kw = dict(color='#2d2d5e', initcolor='none')

    ax_A  = fig.add_axes([0.08, 0.20, 0.44, 0.03])
    ax_nu = fig.add_axes([0.08, 0.15, 0.44, 0.03])
    ax_D  = fig.add_axes([0.08, 0.10, 0.44, 0.03])
    ax_dt = fig.add_axes([0.08, 0.05, 0.44, 0.03])

    sl_A  = Slider(ax_A,  'Activity  A',  1e3, 1e6, valinit=A,   valfmt='%.1e', **sl_kw)
    sl_nu = Slider(ax_nu, 'Alignment ν',  -2.0, 0.0, valinit=0.0, valfmt='%.2f', **sl_kw)
    sl_D  = Slider(ax_D,  'Noise  D',     0.0, 0.02, valinit=2.5e-3, valfmt='%.4f', **sl_kw)
    sl_dt = Slider(ax_dt, 'dt',           0.002, 0.05, valinit=0.02, valfmt='%.3f', **sl_kw)

    for sl in (sl_A, sl_nu, sl_D, sl_dt):
        sl.label.set_color('white')
        sl.valtext.set_color('#aef')

    # ── buttons ──
    btn_kw = dict(color='#2d2d5e', hovercolor='#4d4d9e')

    ax_pause  = fig.add_axes([0.60, 0.34, 0.10, 0.05])
    ax_reset  = fig.add_axes([0.72, 0.34, 0.10, 0.05])
    ax_extens = fig.add_axes([0.60, 0.27, 0.12, 0.05])
    ax_contra = fig.add_axes([0.74, 0.27, 0.12, 0.05])

    btn_pause  = Button(ax_pause,  'Pause',      **btn_kw)
    btn_reset  = Button(ax_reset,  'Reset',      **btn_kw)
    btn_extens = Button(ax_extens, 'Extensile',  **btn_kw)
    btn_contra = Button(ax_contra, 'Contractile',**btn_kw)

    for b in (btn_pause, btn_reset, btn_extens, btn_contra):
        b.label.set_color('white')

    # ── radio: view mode ──
    ax_radio = fig.add_axes([0.60, 0.05, 0.37, 0.20])
    ax_radio.set_facecolor('#0d0d1a')
    radio = RadioButtons(ax_radio, VIEW_MODES,
                         activecolor='#6699ff')
    ax_radio.set_title('View mode', color='#aaa', fontsize=8, pad=2)
    for lbl in radio.labels:
        lbl.set_color('white')
        lbl.set_fontsize(8)

    # ── preset buttons ──
    preset_axes = []
    preset_btns = []
    for i, label in enumerate(PRESETS):
        axi = fig.add_axes([0.04 + i*0.19, 0.22, 0.17, 0.04])
        bi  = Button(axi, label.replace('\n', ' '), **btn_kw)
        bi.label.set_color('#aef')
        bi.label.set_fontsize(7)
        preset_axes.append(axi)
        preset_btns.append(bi)

    # ── initial plot ──
    dummy = np.zeros((N, N))
    im = ax_main.imshow(dummy, origin='lower', extent=[0,1,0,1],
                        cmap='hot', vmin=0, vmax=1,
                        interpolation='bilinear', animated=True)
    cbar = fig.colorbar(im, ax=ax_main, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(colors='white', labelsize=7)
    cbar.outline.set_edgecolor('#444')
    title_txt = ax_main.set_title('', color='white', fontsize=9, pad=3)

    # Streamline quiver (coarse)
    stride = max(1, N // 16)
    xs = np.linspace(0, 1, N//stride)
    ys = np.linspace(0, 1, N//stride)
    XQ, YQ = np.meshgrid(xs, ys)
    quiv = ax_main.quiver(XQ, YQ,
                          np.zeros_like(XQ), np.zeros_like(YQ),
                          color='white', alpha=0.55, scale=None,
                          width=0.003, headwidth=3, headlength=4)

    spec_line, = ax_spec.semilogy([1], [1], color='#6699ff', lw=1.5)
    ax_spec.set_xlim(0, 0.5)

    # ── callbacks ──
    def apply_sliders(_=None):
        sim.A  = sl_A.val
        sim.nu = sl_nu.val
        sim.D  = sl_D.val
        sim.dt = sl_dt.val
        sim._prev_rhs = None   # reset AB2 when params change

    sl_A.on_changed(apply_sliders)
    sl_nu.on_changed(apply_sliders)
    sl_D.on_changed(apply_sliders)
    sl_dt.on_changed(apply_sliders)

    def toggle_pause(_):
        running[0] = not running[0]
        btn_pause.label.set_text('Resume' if not running[0] else 'Pause')
        fig.canvas.draw_idle()

    def do_reset(_):
        sim.reset()
        frame_n[0] = 0

    def set_extensile(_):
        sim.S = +1
        sim._prev_rhs = None

    def set_contractile(_):
        sim.S = -1
        sim._prev_rhs = None

    btn_pause.on_clicked(toggle_pause)
    btn_reset.on_clicked(do_reset)
    btn_extens.on_clicked(set_extensile)
    btn_contra.on_clicked(set_contractile)

    def set_view(label):
        view[0] = label

    radio.on_clicked(set_view)

    def set_preset(pname):
        p = PRESETS[pname]
        sim.S  = p['S']
        sim.nu = p['nu']
        sl_nu.set_val(p['nu'])
        sim._prev_rhs = None
        sim.reset()

    for btn, pname in zip(preset_btns, PRESETS):
        btn.on_clicked(lambda _, n=pname: set_preset(n))

    # ── Frank energy spectrum helper ──
    def frank_spectrum(frank):
        N = frank.shape[0]
        fh  = np.abs(np.fft.fft2(frank))**2
        k1d = np.fft.fftfreq(N)        # in units of 1/L (cycle/box)
        KX_, KY_ = np.meshgrid(k1d, k1d, indexing='ij')
        kr = np.sqrt(KX_**2 + KY_**2)
        kr_flat  = kr.ravel()
        fh_flat  = fh.ravel()
        bins = np.linspace(0, 0.5, N//2)
        spec = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            sel = (kr_flat >= bins[i]) & (kr_flat < bins[i+1])
            spec[i] = fh_flat[sel].mean() if sel.any() else 0.0
        qmid = 0.5*(bins[:-1]+bins[1:])
        return qmid, np.maximum(spec, 1e-30)

    # ── animation ──
    def animate(frame):
        frame_n[0] += 1
        if running[0]:
            sim.step(n_sub=STEPS_PER_FRAME)

        vx, vy = sim.velocity()
        speed  = np.hypot(vx, vy)
        frank  = sim.frank_density()
        v_mode = view[0]

        if v_mode == 'Speed':
            data = speed
            cmap, title = 'viridis', 'Flow speed  |v|'
        elif v_mode == 'Frank energy':
            data = frank
            cmap, title = 'hot', 'Frank energy density  |∇θ|²'
        elif v_mode == 'Director angle':
            data = sim.theta % np.pi    # head-tail symmetry
            cmap, title = 'hsv', 'Director angle  θ  (mod π)'
        elif v_mode == 'Vorticity':
            data = sim.vorticity()
            cmap, title = 'RdBu', 'Vorticity  ω'
        else:
            data = sim.psi
            cmap, title = 'RdBu', 'Stream function  ψ'

        vmin, vmax = data.min(), data.max()
        if abs(vmax - vmin) < 1e-20:
            vmax = vmin + 1.0

        im.set_data(data.T)
        im.set_clim(vmin, vmax)
        im.set_cmap(cmap)
        title_txt.set_text(title)

        # Quiver (coarse velocity)
        stride = max(1, sim.N // 16)
        ux = vx[::stride, ::stride]
        uy = vy[::stride, ::stride]
        spd_sub = speed[::stride, ::stride]
        spd_max = spd_sub.max() + 1e-20
        quiv.set_UVC(ux / spd_max, uy / spd_max)

        # Frank spectrum
        q, sp = frank_spectrum(frank)
        spec_line.set_data(q, sp)
        ax_spec.set_ylim(sp[sp>0].min() * 0.1 if sp.max()>0 else 1e-6,
                         sp.max() * 10)

        # Info text
        regime = ('Strong turb.' if sim.S * sim.nu > 1
                  else 'Arrested' if sim.S * sim.nu < -1
                  else 'Non-aligning')
        info_txt.set_text(
            f't = {sim.t:8.1f}   N = {sim.N}\n'
            f'A = {sim.A:.2e}   R = {sim.R:.1f}\n'
            f'ν = {sim.nu:.2f}   S = {sim.S:+d}   Sν = {sim.S*sim.nu:.2f}\n'
            f'Regime: {regime}\n'
            f'⟨speed⟩ = {speed.mean():.3f}   max = {speed.max():.3f}'
        )

        return im, quiv, spec_line, info_txt, title_txt

    anim = FuncAnimation(fig, animate, interval=60, blit=False, cache_frame_data=False)

    return fig, sim, anim


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    fig, sim, anim = build_ui(N=N)
    plt.show()
