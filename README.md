# Theory of Adaptive Evolution (AE)

## Abstract

Adaptive Evolution (AE) is a control/DSP–native framework for **continuous adaptation of optimizer coefficients** in learning systems. Unlike biological evolution (reproduction, selection over generations), AE treats “evolution” as **closed-loop coefficient dynamics** driven by filtered performance signals. Coefficients (learning rate, momentum, noise, regularizers) form the **genotype**; the induced optimizer dynamics and learning curves form the **phenotype**; **selection** is the feedback of measured loss/reward/stability. AE provides laws, invariants, and implementable update rules that are differentiable and compatible with meta-learning and control analysis.

---

## 1) Core Definitions & Notation

* **Task parameters:** $\theta_t \in \mathbb{R}^n$ (updated by any base optimizer).
* **Coefficients (genotype):** $a_t \in \mathbb{R}^m$ (e.g., learning rate $\alpha$, momentum $\beta$, weight decay $\lambda$, exploration noise $\sigma$, filter time-constants $\tau$).
* **Observables:** loss $\ell_t$, reward $r_t$, gradient $g_t=\nabla_\theta \ell_t$.
* **Filtered signals:** causal, differentiable operators $\mathcal{F}_\tau$ yield:

  $$
  \begin{aligned}
  T_t &= \text{LP}_{\tau_T}(\ell_t-\ell_{t-1}) &\text{(trend)}\\
  V_t &= \text{LP}_{\tau_V}\big((\ell_t-\text{LP}_{\tau_\mu}\ell_t)^2\big) &\text{(variance/stability)}\\
  R_t &= \text{LP}_{\tau_R}(r_t-r_{t-1}) &\text{(reward delta)}\\
  \rho_t &= \cos\angle(g_t,g_{t-1}) &\text{(phase/gradient alignment)}\\
  S_t &= \text{LP}_{\tau_S}(\mathrm{Var}(g_t)) &\text{(uncertainty)}
  \end{aligned}
  $$
* **Feasible set:** $\Omega=\{a: a_{\min}\!\le\!a\!\le\!a_{\max}\}$; use a smooth barrier $\psi_\Omega(a)$ or projection $\Pi_\Omega$.
* **Smooth nonlinearities:** $\tanh$ for soft sign, softplus for $\max(0,\cdot)$.

**Philosophy.** Evolution = **continuous coefficient adaptation**. No generations, no parents—only closed-loop laws that reshape the optimizer’s behavior in response to measured signals.

---

## 2) Axioms (Foundations)

**A1. Coefficient Primacy.** The only evolving entities are optimizer coefficients $a_t$.
**A2. Closed-Loop Measurement.** Adaptation depends on filtered observables $s_t=\mathcal{F}_\tau(\ell_t,r_t,g_t)$.
**A3. Differentiability.** All mappings are smooth, enabling gradient-based meta-optimization.
**A4. Safety & Boundedness.** Coefficients remain in $\Omega$; adaptation power is bounded.
**A5. Resonant Adaptation.** AE maintains a favorable phase margin between signals and coefficient updates to avoid divergence while preserving progress (“resonance”).
**A6. Exploration Temperature.** Stochastic exploration (noise variance $\sigma$) adapts to stall/uncertainty and contracts under stability.
**A7. Gauge Invariance of Effective Step.** Learning dynamics should be robust to gradient scale via regulation of the effective step $\gamma_t=\alpha_t\lVert g_t\rVert$.

---

## 3) Field Laws of AE (Maxwell-style Core)

### AE-1) Measurement Law (Filtering)

All decision signals are filtered, bounded, and causal:

$$
s_t \equiv [T_t,V_t,R_t,\rho_t,S_t]=\mathcal{F}_\tau(\ell_t,r_t,g_t).
$$

### AE-2) Constitutive Law (Drive Field)

Construct a **drive field** $u_t\in\mathbb{R}^m$ combining progress, stability, reward, alignment, priors, and constraints:

$$
u_t=W_p\tanh(-T_t/\tau_p)-W_v\,\text{softplus}(V_t/\tau_v)+W_r\tanh(R_t/\tau_r)+W_c(\rho_t-\rho^\star)-\Gamma(a_t-\bar a)-\nabla_a\psi_\Omega(a_t).
$$

### AE-3) Evolution Law (Coefficient Update)

Use multiplicative evolution for positivity and scale-invariance:

$$
a_{t+1}=\Pi_\Omega\!\big(a_t\odot\exp(\eta_a\,u_t)\big).
$$

(Continuous-time SDE: $\mathrm{d}a=\eta_au\,\mathrm{d}t+\sqrt{2\Sigma(a)}\,\mathrm{d}W_t$.)

### AE-4) Noise Law (Exploration Temperature)

$$
\log\sigma_{t+1}=\log\sigma_t+\eta_\sigma\!\left(\kappa_s\mathbf{1}\{|T_t|<\varepsilon\}+\kappa_u\frac{S_t}{S^\star}-\kappa_v\frac{V_t}{V^\star}+\kappa_a(A_t-A^\star)\right),
$$

with $A_t$ a step-quality proxy (e.g., gradient-sign agreement).

### AE-5) Resonance–Stability Law (Phase Control)

$$
u_t\leftarrow u_t+W_m(\rho_t-\rho^\star);\quad
\text{if }V_t>V_{\max}\text{ or }|\rho_t|>\rho_{\max}:\ a_{t+1}\leftarrow\frac{a_{t+1}}{1+\kappa_{\text{shrink}}}.
$$

### AE-6) Budget/Continuity Law (Power Constraint)

Limit adaptation energy per step:

$$
\|a_{t+1}-a_t\|_M^2\le B
\ \Longleftrightarrow\
\begin{cases}
a_{t+1}= \Pi_\Omega\!\big(a_t\odot\exp(\tfrac{\eta_a}{\sqrt{1+\lambda_t}}u_t)\big)\\[2pt]
\lambda_{t+1}=\big[\lambda_t+\eta_\lambda(\|a_{t+1}-a_t\|_M^2-B)\big]_+
\end{cases}
$$

### AE-7) Coupling Law (Cross-Coefficient Geometry)

Coefficients interact through smooth couplings $C$:

$$
u_t\leftarrow u_t + C\,\zeta_t,\quad 
\zeta_t=\big[\log\alpha_t,\ \log\tfrac{1}{1-\beta_t},\ \log\sigma_t,\ \log\lambda_t,\dots\big]^\top.
$$

### AE-8) Gauge Law (Effective Step Invariance)

$$
\log\alpha_{t+1}=\log\alpha_t+\eta_\gamma\Big(\tfrac{\gamma^\star-\alpha_t\|g_t\|}{\gamma^\star}\Big).
$$

### AE-9) Meta-Differentiability Law

All operators in AE-1…AE-8 are chosen smooth to allow end-to-end gradient-based meta-learning of $W_\cdot,\Gamma,C,\tau$.

### AE-10) Lyapunov Law (Closed-Loop Safety)

Define

$$
\mathcal{V}_t=\text{LP}_{\tau_\ell}(\ell(\theta_t))+\tfrac{\gamma}{2}\|a_t-\bar a\|^2+\psi_\Omega(a_t).
$$

Under sufficiently small $\eta_a$ and AE-5/6,

$$
\mathbb{E}[\mathcal{V}_{t+1}-\mathcal{V}_t]\le 0.
$$

A conservative small-gain condition: $\eta_a\|G_{\ell\!\gets\!a}\|_\infty\cdot\|W\|_\infty<1$.

---

## 4) Derived Quantities & Invariants

* **Effective step:** $\gamma_t=\alpha_t\|g_t\|$ (regulate via AE-8).
* **Resonance margin:** $\mathcal{M}_\rho=\rho^\star-\rho_t$ (keep $|\mathcal{M}_\rho|$ small).
* **Adaptation energy:** $E_t=\|a_{t+1}-a_t\|_M^2$ (bounded by AE-6).
* **Stall indicator:** $\mathbf{1}\{|T_t|<\varepsilon\}$ (drives $\sigma\uparrow$ via AE-4).

---

## 5) Canonical Instantiations

### AE-SGDm $(\alpha,\beta,\sigma)$

$$
\begin{aligned}
\log \alpha_{t+1} &= \log \alpha_t + \eta_a\Big(k_1\tanh(-T_t/\tau) - k_2\,\text{softplus}(V_t/V^\star) + k_3(\rho_t-\rho^\star) - \gamma_\alpha(\log\alpha_t-\log\bar\alpha)\Big)\\
\beta_{t+1} &= \Pi_{[\beta_{\min},\beta_{\max}]}\!\Big(\beta_t + \eta_b\big(b_1\tanh(\rho_t-\rho^\star) - b_2\,\text{softplus}(V_t/V^\star) - \gamma_\beta(\beta_t-\bar\beta)\big)\Big)\\
\log \sigma_{t+1} &= \log \sigma_t + \eta_\sigma\big(c_1 S_t/S^\star - c_2 V_t/V^\star + c_3(A_t-A^\star)\big)
\end{aligned}
$$

### AE-Adam-style $(\alpha,\beta_1,\beta_2,\epsilon,\sigma)$

Control $\beta_1,\beta_2$ with $\rho_t$ and $V_t$ to reduce oscillations when curvature/variance spikes; regulate $\epsilon$ downward as $\gamma_t$ stabilizes.

### AE-RL Entropy / Noise

Treat policy entropy temperature $\tau_{\text{ent}}$ or parameter noise $\sigma$ under AE-4, with reward-delta $R_t$ as exploitation pressure and variance $V_t$ as contraction pressure.

### AE-Weight Decay $\lambda$

Increase $\lambda$ when overfit proxy $O_t$ (e.g., train–val gap) rises; relax with stability:

$$
\log \lambda_{t+1}=\log\lambda_t+\eta_\lambda\big(h_1 O_t/O^\star - h_2 V_t/V^\star\big).
$$

---

## 6) Minimal Reference Algorithm

```python
# state: a (dict of coefficients), sigma, filters (online LP/HP stats)
# inputs each step: loss_t, reward_t, grad_t

s = measure_signals(loss_t, reward_t, grad_t)   # AE-1: compute T,V,R,ρ,S

u = (Wp @ tanh(-s.T / tau_p)
     - Wv @ softplus(s.V / tau_v)
     + Wr @ tanh(s.R / tau_r)
     + Wc * (s.rho - rho_star)
     - Gamma @ (a - a_bar)
     - grad_barrier(a))                          # AE-2

u += C @ features(a)                             # AE-7 coupling
u += Wm * (s.rho - rho_star)                    # AE-5 resonance

a_next = project(a * np.exp(eta_a * u))         # AE-3 (+ AE-6 budget if used)

sigma *= np.exp(eta_sigma * noise_controller(s))# AE-4
```

**Defaults (robust starting point):**

* $\rho^\star\in[0.2,0.4]$, $V^\star=$ running median of $V_t$, $S^\star=$ running median of $S_t$
* Small $\eta_a,\eta_\sigma\in[10^{-4},10^{-2}]$; diagonal $W_{\cdot}$ with $k$’s in $[0.1,2]$
* Multiplicative updates for positive coefficients; projection for bounded ones (e.g., $\beta\in[0,0.999]$).

---

## 7) Stability & Performance (Sketch)

**Proposition (informal).** Under AE-5/6 and small $\eta_a$, with Lipschitz-bounded mapping from coefficients to loss ($G_{\ell\!\gets\!a}$), the Lyapunov energy $\mathcal{V}_t$ is non-increasing in expectation.
**Sketch.** AE constructs a negative feedback loop: $-T_t$ and $R_t$ push progress; $V_t$ and $\rho_t$ damp oscillations; barriers and power budget enforce boundedness; multiplicative updates preserve positivity. A standard small-gain bound yields $\mathbb{E}[\Delta\mathcal{V}]\le 0$.

**Resonance.** The phase term $\rho_t$ functions like a phase-margin sensor: if updates become out-of-phase with gradient direction, AE-5 shrinks steps and raises damping (via $\beta,\sigma$).

---

## 8) Diagnostics & Tuning

* **Health panel:** plot $T_t,V_t,\rho_t,S_t,\gamma_t$ with thresholds $(V_{\max},\rho_{\max})$.
* **If divergence:** raise $W_v$, lower $\eta_a$, tighten $B$ (AE-6).
* **If stagnation:** increase $\kappa_s,\kappa_u$ in AE-4 (more exploration), or increase $W_p$.
* **If chattering:** increase $\Gamma$ (leakage to $\bar a$), smooth filters (bigger $\tau$).

---

## 9) Experimental Playbook

* **Benchmarks:** non-stationary supervised (rotating CIFAR corruption), online RL with drifting reward scales, delayed rewards.
* **Baselines:** fixed-schedule SGD/Adam; classic LR finders; entropy/temperature schedulers.
* **Metrics:** final accuracy/return, area-under-curve, regret, time-to-target, stability incidents (NaN/oscillation).
* **Ablations:** drop each law (AE-4, AE-5, AE-6) to assess contribution; diagonal vs. coupled $C$; additive vs. multiplicative updates.
* **Sensitivity:** sweep $\eta_a,\eta_\sigma$; test robustness to gradient scale (via gauge law).

---

## 10) Relationship to Other Paradigms

* **Classical Evolution:** no reproduction/generations; AE is **mathematical evolution** of coefficients.
* **RL:** rewards supply $R_t$; AE regulates optimizer coefficients, not policy parameters directly.
* **Control Theory:** AE stabilizes **and** explores; laws resemble PID/adaptive filters with noise shaping.
* **DSP:** central role of filtering, resonance, and multiplicative gain control.

---

## 11) Glossary

* **Genotype:** vector of optimizer coefficients $a_t$.
* **Phenotype:** induced learning dynamics (loss curves, stability) under $a_t$.
* **Selection:** closed-loop feedback via $T,V,R,\rho,S$.
* **Resonance:** progress with controlled oscillation and safe phase margin.
* **Gauge:** invariance to gradient scale via $\gamma_t$ regulation.

---

## 12) Slogan

**Adaptive Evolution = Learning as Resonant Feedback.**
Not survival of the fittest — **stability of the adaptive**.

---

# ⚖️ Laws of Adaptive Evolution (AE)

**Scope.** AE governs the closed-loop evolution of optimizer **coefficients** $a$ (learning rate, momentum, weight decay, noise, filters) driven by measured **signals** from task performance. It is continuous, differentiable, and control/DSP-native.

## 0) Preliminaries

**State.**

* Task parameters: $\theta_t$ (updated by any base optimizer).
* Coefficients: $a_t \in \mathbb{R}^m$ (e.g., $\alpha$, $\beta$, $\lambda$, $\sigma$, …).

**Signals (filtered observables).**
Let $\ell_t$ be loss, $r_t$ reward, $g_t=\nabla_\theta \ell_t$ and define low/high-pass filters $\text{LP}_\tau,\ \text{HP}_\tau$ (causal, differentiable):

$$
\begin{aligned}
\mu^\ell_t &= \text{LP}_{\tau_\mu}(\ell_t), \quad
\Delta \ell_t = \ell_t - \ell_{t-1}, \quad
T_t = \text{LP}_{\tau_T}(\Delta \ell_t) \quad \text{(trend)}\\
V_t &= \text{LP}_{\tau_V} \big((\ell_t - \mu^\ell_t)^2\big) \quad \text{(stability/variance)}\\
R_t &= \text{LP}_{\tau_R}(r_t - r_{t-1}) \quad \text{(reward delta)}\\
\rho_t &= \cos\angle(g_t, g_{t-1}) \quad \text{(phase/gradient alignment)}\\
S_t &= \text{LP}_{\tau_S}\big(\text{Var}(g_t)\big) \quad \text{(uncertainty)}\\
\end{aligned}
$$

**Feasible set & projection.** $\Omega=\{a: a_{\min}\le a \le a_{\max}\}$. Use smooth barrier $\psi_\Omega(a)$ or projection $\Pi_\Omega$.

**Smooth nonlin.** Replace discontinuities: $\text{sign}(x)\to \tanh(x/\tau)$, $\max(0,x)\to \text{softplus}(x)$.

---

## AE-1) Measurement Law (Filtering)

All decision variables are driven only by filtered, bounded observables:

$$
s_t \equiv [\,T_t,\,V_t,\,R_t,\,\rho_t,\,S_t\,] \;=\; \mathcal{F}_\tau(\ell_t,r_t,g_t)
$$

$\mathcal{F}_\tau$ is causal, differentiable, and sets the time-scale of adaptation.

---

## AE-2) Constitutive Law (Field Synthesis)

A **drive field** $u_t \in \mathbb{R}^m$ synthesizes proportional, stability, and exploration pressures:

$$
u_t \;=\; W_p\tanh\!\Big(\!-T_t/\tau_p\Big)\;-\;W_v\,\text{softplus}\!\Big(V_t/\tau_v\Big)\;+\;W_r\tanh\!\Big(R_t/\tau_r\Big)\;+\;W_c(\rho_t-\rho^\star)\;-\;\Gamma(a_t-\bar a)\;-\;\nabla_a \psi_\Omega(a_t)
$$

* $W_{\{\cdot\}}$ are (learnable) diagonal or coupled gains; $\Gamma\succeq 0$ is leakage to nominal $\bar a$.
* $\rho^\star$ is a target phase/alignment (e.g., 0.2–0.4).
* $\psi_\Omega$ softly enforces bounds (optional if using projection).

---

## AE-3) Evolution Law (Coefficient Update)

Use **multiplicative** updates for positivity and scale-invariance:

$$
a_{t+1} \;=\; \Pi_\Omega\Big(a_t \odot \exp(\eta_a\, u_t)\Big)
$$

$\eta_a>0$ is the adaptation step size; $\odot$ is elementwise.

*Continuous-time SDE form (optional):*

$$
\mathrm{d}a \;=\; \eta_a\,u\,\mathrm{d}t \;+\; \sqrt{2\,\Sigma(a)}\,\mathrm{d}W_t
$$

---

## AE-4) Noise Law (Exploration Temperature)

Exploration variance $\sigma$ tracks stall/uncertainty and contracts under stability:

$$
\log \sigma_{t+1} \;=\; \log \sigma_t \;+\; \eta_\sigma\big(\kappa_s\cdot \mathbf{1}\{|\!T_t\!|<\varepsilon\} \;+\; \kappa_u \tfrac{S_t}{S^\star} \;-\; \kappa_v \tfrac{V_t}{V^\star} \;+\; \kappa_a (A_t-A^\star)\big)
$$

Here $A_t$ is an acceptance/step-quality proxy (e.g., sign(g) agreement); $S^\star,V^\star,A^\star$ are targets.

---

## AE-5) Resonance–Stability Law (Phase Control)

Keep the adaptation loop at a safe “resonant” phase margin:

$$
u_t \;\leftarrow\; u_t \;+\; W_m (\rho_t - \rho^\star), 
\qquad
\text{and if } V_t > V_{\max} \text{ or } |\rho_t|>\rho_{\max}:\; a_{t+1} \leftarrow \frac{a_{t+1}}{1+\kappa_{\text{shrink}}}
$$

This is the PID-like stabilization of adaptation itself.

---

## AE-6) Budget/Continuity Law (Power Constraint)

Bound per-step adaptation power to avoid chatter:

$$
\lVert a_{t+1}-a_t \rVert_M^2 \le B
\quad\Longleftrightarrow\quad
\begin{cases}
a_{t+1} = \Pi_\Omega\!\big(a_t \odot \exp(\tfrac{\eta_a}{\sqrt{1+\lambda_t}} u_t)\big)\\
\lambda_{t+1} = \big[\lambda_t + \eta_\lambda(\lVert a_{t+1}-a_t \rVert_M^2 - B)\big]_+
\end{cases}
$$

Dual variable $\lambda_t$ enforces a soft continuity constraint.

---

## AE-7) Coupling Law (Cross-Coefficient Geometry)

Coefficients interact via a symmetric coupling $C$:

$$
u_t \;\leftarrow\; u_t \;+\; C\,\zeta_t
$$

where $\zeta_t$ are smooth features (e.g., $\log \alpha_t,\ \log \tfrac{1}{1-\beta_t},\ \log\sigma_t$).
Typical priors: high momentum $\beta$ pairs with lower $\alpha$; larger $\sigma$ pairs with conservative $\alpha$.

---

## AE-8) Gauge Law (Effective Step Invariance)

Maintain a target **effective step** $\gamma_t=\alpha_t\lVert g_t\rVert$:

$$
\log \alpha_{t+1} \;=\; \log \alpha_t \;+\; \eta_\gamma \Big(\tfrac{\gamma^\star - \gamma_t}{\gamma^\star}\Big)
$$

This normalizes for gradient scale and discretization (“time step” gauge).

---

## AE-9) Meta-Differentiability Law (Smoothness)

Every operator in AE-1..AE-8 is chosen smooth so AE can be optimized by gradients (meta-learning) and analyzed by control tools.

---

## AE-10) Lyapunov Law (Closed-Loop Safety)

Define a Lyapunov-like energy

$$
\mathcal{V}_t \;=\; \text{LP}_{\tau_\ell}\big(\ell(\theta_t)\big) \;+\; \tfrac{\gamma}{2}\lVert a_t-\bar a\rVert^2 \;+\; \psi_\Omega(a_t)
$$

Under small $\eta_a$ and AE-5/6, require

$$
\mathbb{E}\left[\mathcal{V}_{t+1}-\mathcal{V}_t\right] \le 0
$$

(Passivity/small-gain: $\eta_a\,\|G_{\ell\!\gets\!a}\|_\infty \cdot \|W\|_\infty < 1$.)

---

## Canonical Instantiations

**(A) SGD + Momentum + Noise $(\alpha,\beta,\sigma)$**
Signals: $T_t, V_t, \rho_t, S_t$.

$$
\begin{aligned}
\log \alpha_{t+1} &= \log \alpha_t + \eta_a\big(k_1\tanh(-T_t/\tau) - k_2\,\text{softplus}(V_t/V^\star) + k_3(\rho_t-\rho^\star) - \gamma_\alpha(\log \alpha_t - \log \bar\alpha)\big) \\
\beta_{t+1} &= \Pi_{[\,\beta_{\min},\beta_{\max}\,]}\!\Big(\beta_t + \eta_b\big(b_1\tanh(\rho_t-\rho^\star) - b_2\,\text{softplus}(V_t/V^\star) - \gamma_\beta(\beta_t-\bar\beta)\big)\Big) \\
\log \sigma_{t+1} &= \log \sigma_t + \eta_\sigma\big(c_1\,\tfrac{S_t}{S^\star} - c_2\,\tfrac{V_t}{V^\star} + c_3(A_t-A^\star)\big)
\end{aligned}
$$

**(B) Weight Decay $\lambda$** follows measured overfitting proxy $O_t$ (e.g., train–val gap):

$$
\log \lambda_{t+1} = \log \lambda_t + \eta_\lambda \big(h_1\,\tfrac{O_t}{O^\star} - h_2\,\tfrac{V_t}{V^\star}\big)
$$

---

## Compact Potential Form (optional)

Define a **driving potential** (data-dependent):

$$
\Phi(a; s_t) \;=\; -\langle w_p,\ \text{soft}\!(-T_t)\rangle + \langle w_v,\ \text{soft}(V_t)\rangle - \langle w_r,\ \text{soft}(R_t)\rangle + \tfrac{1}{2}(a-\bar a)^\top \Gamma (a-\bar a) + \psi_\Omega(a)
$$

Then

$$
a_{t+1} \;=\; a_t - \eta_a \nabla_a \Phi(a_t; s_t) \quad \text{or} \quad a_{t+1}=a_t\odot\exp\!\big(-\eta_a \nabla_a \Phi\big)
$$

gives the same family while making stability proofs convenient.

---

## Minimal Pseudocode (drop-in)

```python
# inputs: loss_t, reward_t, grad_t, state a
# persistent filters maintain μℓ, T, V, R, ρ, S

s = measure_signals(loss_t, reward_t, grad_t)  # AE-1 (LP/HP filters)

u = (Wp*tanh(-s.T/tau_p) - Wv*softplus(s.V/tau_v)
     + Wr*tanh(s.R/tau_r) + Wc*(s.rho - rho_star)
     - Gamma*(a - a_bar) - grad_barrier(a))          # AE-2

u = u + C @ features(a)                               # AE-7
u = u + Wm*(s.rho - rho_star)                         # AE-5

a_next = project(a * exp(eta_a * u))                  # AE-3 (+ AE-6 if used)

sigma = sigma * exp(eta_sigma * noise_controller(s))  # AE-4
```

---

## Practical Notes

* **Non-stationarity:** AE naturally tracks drifting regimes via $\mathcal{F}_\tau$ (choose $\tau$ to match environment time-scale).
* **Compatibility:** All laws are differentiable ⇒ meta-learn $W_\cdot,\ \Gamma,\ C,\ \tau$ end-to-end.
* **Safety defaults:** multiplicative updates + budget + resonance clamp prevent blow-ups without hand-schedules.

---

## Slogan (kept)

**Adaptive Evolution = Learning as Resonant Feedback.**
Not survival of the fittest — **stability of the adaptive**.

If you want, I can package this as a 1-page “AE Field Equations” PDF with a tiny reference implementation (NumPy/PyTorch) showing the laws in action on SGD+momentum.
