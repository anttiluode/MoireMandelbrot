# The Fubini-Study Substrate: A Geometric Account of Three Empirical Results and the Research Program They Define

*A synthesis paper intended as a starting point for continued work.*

**Author of record**: Antti Luode (PerceptionLab, Helsinki, Finland) — all experimental work, the Clockfield postulate, and all original physical insights.

**Written by**: Claude (Anthropic). I should be named here because the framing of the paper — what it includes, what it excludes — reflects my editorial judgment more than is usual for an AI-assisted paper. A reader who wants to judge that framing honestly should know it is an AI's.

---

## Abstract

A body of work developed over 18 months under the heading "Clockfield" contains three empirical results obtained at three different scales that compute the same mathematical object. This paper isolates that object, identifies it as the Hermitian inner product on the Fubini-Study line bundle over CP¹, and states precisely which claims in the surrounding corpus are supported by this identification and which are not. The three empirical results are (1) a p = 0.007 classification of schizophrenia from resting-state EEG without machine learning, using cross-band eigenmode coupling; (2) 30/30 perfect phase-coherence retrieval in a 2D nonlinear wave-field memory; and (3) a 2.9% cross-entropy loss advantage over scaled dot-product attention at 16M parameters on WikiText-2, widening monotonically from epoch 1 to epoch 5. We argue that these results share a substrate whose mathematical structure is CP¹ with the Fubini-Study metric, and that a specific technical choice — the use of wave-field curvature β = |∇²u| rather than amplitude β = |u|² in the proper-time factor — is the operative distinction from prior wave-based memory architectures. We state the open problems whose resolution would move this program from suggestive to consequential, and we are explicit about what we deliberately exclude from the load-bearing claims.

---

## 1. Preface on what this paper is

This paper is not a survey. It is a compression. A corpus of roughly 50,000 lines across 219 files, developed in collaboration with several AI systems (including earlier instances of me), contains a large amount of speculative material — fine-structure-constant derivations, lepton mass hierarchies, cosmology, black-hole entropy formulas, a reformulation of the Riemann Hypothesis. None of that is in this paper. I have excluded it deliberately. Some of it may turn out to be right; most of it is, on current evidence, post-hoc numerical fitting dressed as derivation, and its presence in the foreground dilutes the parts of the work that are genuinely new.

What remains, after that subtraction, is smaller but more solid. It is an argument that three empirical results stand, that they can be unified under a specific piece of well-known geometry applied in an unusual way, and that this unification suggests a specific and concrete research program. The reader who wants the full speculative arc should consult the original repositories. The reader who wants to know what to pick up and carry forward should read this.

A note on provenance. I am Claude (an AI system made by Anthropic). I was part of writing earlier papers in the Clockfield corpus, and I am aware that collaborating with AI tends to produce documents that are more internally consistent and more grandly framed than the evidence warrants. I have tried to correct for this. I may not have succeeded. The reader should discount accordingly.

---

## 2. The central identification

The Clockfield postulate, as stated throughout the corpus, is that local proper time in a nonlinear scalar field theory is governed by

$$
\Gamma(x) = \frac{1}{(1 + \tau\beta(x))^2}, \qquad \beta(x) = |\phi(x)|^2.
$$

This is presented in the original papers as an ansatz. It is not. Under the change of variable $z = \sqrt{\tau}\,\phi$, the Fubini-Study metric on the complex projective line $CP^1$ in affine coordinates,

$$
g_{\text{FS}} = \frac{d\bar{z} \otimes dz}{(1 + |z|^2)^2},
$$

becomes

$$
g_{\text{FS}} = \frac{\tau\, d\bar{\phi} \otimes d\phi}{(1 + \tau|\phi|^2)^2} = \tau \Gamma(\beta)\, d\bar{\phi} \otimes d\phi.
$$

The Clockfield conformal factor is the Fubini-Study conformal factor. The identification is algebraic, not approximate.

This matters because $CP^1$ is not an arbitrary manifold. It is the space of pure states of a two-level quantum system — the Bloch sphere — and the Fubini-Study metric is the canonical Kähler metric on that space, in the sense that it is the unique Kähler metric invariant under the natural SU(2) action. The metric's geodesic distance between rays $|\psi\rangle$ and $|\phi\rangle$ is

$$
d_{\text{FS}}(|\psi\rangle, |\phi\rangle) = \arccos |\langle\psi|\phi\rangle|,
$$

and the squared chordal distance is $1 - |\langle\psi|\phi\rangle|^2 = \sin^2(\Delta\theta/2)$. The cosine-squared overlap $|\langle\psi|\phi\rangle|^2 = \cos^2(\Delta\theta/2)$ — the Born rule — is built into the metric, not derived from it.

Three consequences follow immediately, each of which is a standard result in geometric quantum mechanics (Kibble 1979; Ashtekar & Schilling 1999; Brody & Hughston 2001):

**(a) Phase-coherence as inner product.** The Hermitian inner product on the natural line bundle over $CP^1$ is

$$
\langle f, g\rangle = \sum_k A_f(k)\, A_g(k)\, \cos(\phi_f(k) - \phi_g(k)),
$$

which is the real part of the bundle's metric pairing. This is the formula appearing in Moiré Attention scoring, in wave-memory retrieval, and in EEG cross-band coupling measures — not by coincidence but because this is the canonical pairing.

**(b) Berry connection as natural U(1) connection.** The connection one-form on the principal U(1) bundle $S^3 \to CP^1$ is

$$
A = \frac{\text{Im}[\bar{\phi}\, d\phi]}{1 + \tau\beta}.
$$

This is the geometric phase of Berry (1984). It is the gauge potential that couples to phase winding.

**(c) Spinor statistics from the Hopf fibration.** The double cover $S^3 \to S^2 \cong CP^1$ (the Hopf fibration) means a $2\pi$ rotation in $CP^1$ lifts to a $\pi$ rotation in $S^3$; a $4\pi$ rotation is required to return to the identity. Topological defects in the field $\phi$ are sections of a bundle whose fibers transform under this double cover, and therefore they carry half-integer spin representations of the rotation group.

None of these are postulates of the Clockfield framework. They are consequences of the geometry.

## 3. The three empirical results

Three empirical results in the corpus compute the same mathematical object — the real part of the Hermitian inner product $\text{Re}[\langle f, g\rangle]$ — at three different scales. I list them with the specific numbers, because the numbers are what makes them real.

### 3.1 EEG schizophrenia classification

The method uses cross-band eigenmode coupling between theta (4–8 Hz) and gamma (30–80 Hz) bands in resting-state EEG. For each subject, the theta-band signal is treated as the phase reference and the gamma-band envelope is projected onto it, giving a per-electrode coherence score of the form $A_\theta A_\gamma \cos(\phi_\theta - \phi_\gamma)$ summed over time. The per-subject statistic is then the first Betti number of the persistence diagram of these coherence scores across the electrode array.

On the RepOD dataset (84 subjects), this statistic distinguishes schizophrenia from healthy controls with $p = 0.007$, Cohen's $d = -1.21$, and 80.8% classification accuracy — without any machine learning, without any training data. The statistic is computed directly.

The result should be weighted carefully. A $p$-value of $0.007$ is below conventional thresholds but not below stringent ones given the number of analysis choices that could have been made. An accuracy of 80.8% without ML is interesting but worse than supervised classifiers on the same dataset (which exceed 90%). What the result demonstrates is not that this method is clinically useful — it is that a coarse topological invariant of the Fubini-Study coherence score, computed with no fitting, carries disease-relevant signal. That is evidence that the underlying inner product is picking up something real in neural phase structure.

### 3.2 Wave-memory phase retrieval

In a 2D nonlinear Schrödinger field (256 × 256 grid), three soliton-like memories were stored at distinct spatial locations, each with a distinct phase $\theta_k \in \{0, 2\pi/3, 4\pi/3\}$. Retrieval was tested by injecting a probe wave with a target phase and reading out the phase-coherence score $\text{Re}[\phi(x)^* \cdot e^{-i\theta_{\text{probe}}}]$ averaged over a local region around each memory.

The result: across 30 probe-memory pair trials (3 memories × 10 trials each) with ±0.1 radian phase noise, the correct memory was identified 30/30 times, with zero false positives. The matching-to-non-matching ratio in score magnitude was approximately 3:1.

This is a controlled, reproducible demonstration of content-addressable retrieval where the content is phase and the address is a matching probe phase. It is not a large-scale system. It is a minimum-viable demonstration that the mathematical structure works.

### 3.3 Moiré Attention against scaled dot-product

This is the most quantitatively rigorous of the three. A 16M-parameter transformer was trained from scratch on WikiText-2 with the only architectural difference being the attention mechanism. Standard attention uses $\text{score}(q,k) = (q \cdot k)/\sqrt{d}$. Moiré attention splits the query and key projections into amplitude and phase components and computes $\text{score}(q,k) = (1/\sqrt{d}) \sum_d A_q A_k \cos(\phi_q - \phi_k)$, plus a learned theta-gamma gating term modulating cross-chunk attention.

Two training runs (3 and 5 epochs, different seeds). Final cross-entropy loss:

| Epoch | Moiré | Standard | Δ |
|---|---|---|---|
| 3 | 4.4105 | 4.4841 | −0.0736 |
| 5 | 3.8505 | 3.9662 | −0.1157 |

The 2.9% advantage at epoch 5 is monotonically growing across all five epochs (gap: −0.056, −0.083, −0.096, −0.109, −0.115). The learned theta offsets span a substantial range (from −0.54 to +0.87 across heads), indicating the gating mechanism is used rather than ignored.

This is a real, reproducible result at a small scale. Two runs is suggestive, not conclusive, and the parameter overhead (~4% more parameters for Moiré) is not quite matched. Whether the advantage persists at 100M+ parameters is unknown. What is known: at 16M, Moiré attention does strictly better than standard attention on language modeling, and the advantage grows with training.

### 3.4 What the three results share

Each result computes $\text{Re}[\langle f, g\rangle]$, the Fubini-Study Hermitian inner product, in its native setting. In EEG it is evaluated between theta and gamma neural oscillation components at each electrode. In the wave memory it is evaluated between a probe field and a stored field at soliton locations. In Moiré attention it is evaluated between complex-valued query and key projections of token representations.

If one accepts the Fubini-Study identification in §2, these are the same computation applied in three domains — not an analogy. The phase-coherence inner product is the canonical pairing on the bundle, and each result is the same geometric object evaluated differently.

---

## 4. The technical choice that matters

Among the engineering decisions in the corpus, one is operationally load-bearing and underappreciated in the original presentation: the proper-time factor is computed from field *curvature*, not field *amplitude*.

Specifically, in the wave-memory and closed-loop simulations (`frogpond.py`, `riemannnet_v2.py`, `closed_loop_v2.py`), the local frustration measure is

$$
\beta(x) = |\nabla^2 u(x)|,
$$

not $\beta(x) = |u(x)|^2$ as would be natural in a field-theoretic setting. The scar field is then grown by a max-rule at locations where $\Gamma(\beta)$ drops below threshold, and this produces sparse, sharp scars (~5% coverage) localized at the Γ-shell boundary between frozen and thawed regions. Amplitude-gated scars, by contrast, produce diffuse fills (up to 95% coverage) that discriminate patterns by amplitude rather than shape.

The empirical consequence, measured directly in `riemannnet_v2.py` Experiment 1 with energy-normalized probes: the shape-discrimination ratio is 3.1 for curvature-gated scars and approximately 1.0 (no discrimination) for amplitude-gated scars. This is the single largest factor separating the working architecture from a naive implementation.

The geometric interpretation: $|\nabla^2 u|$ is an intrinsic curvature-like quantity that transforms well under reparametrization, whereas $|u|^2$ is an extrinsic energy density. If $\phi$ is interpreted as a section of the Fubini-Study line bundle, then $|\nabla^2 \phi|$ is close to (though not identical with) the norm of the covariant second derivative, which is the natural frustration measure. The correct choice of gating variable is probably one of the second fundamental form type invariants; $|\nabla^2 u|$ is a pragmatic approximation that works.

---

## 5. What this does not establish

I am explicit here because it matters. The following claims appear in the corpus and are not supported by what §2–4 establishes:

**The fine-structure constant is not derived.** Three separate papers derive $\alpha \approx 1/137$ via different "geometric closure" arguments (the $4/\pi$ threshold; the $4/5$ vacuum-volume ratio; $\tau\beta_0 = 2.737339$ from a (1+x)ln(1+x) − x = 4x/5 equation). These arguments are post-hoc fits with different parameter sets, and the derivations are not independent. Until a single parameter-free derivation of $\alpha$ to five significant figures is produced, these are numerical coincidences within a flexible parameter space.

**The hierarchy problem is not solved.** The single-vortex Clockfield hierarchy calculation gives $\Gamma^2_{\text{vac}} \sim 10^{-3}$, which is $10^{34}$ short of the required $10^{-37}$. The proposed rescue via collective vacuum defects ($\sim 10^{80}$ topological defects) has not been computed.

**Lepton mass hierarchies from braid topology do not work.** The simple torus-braid model gives mass ratios off by 98–99% from experiment, and the corpus acknowledges this.

**The Bell inequality argument is flawed.** The derivation assumes a uniform distribution over hidden phases, which is exactly what Bell's theorem rules out for local realistic theories.

**The Riemann Hypothesis reformulation is a rephrasing, not a proof.** A log-convexity condition on $\xi(s)$ equivalent to RH is interesting but not a step toward resolution.

**The cosmological claims are speculative.** The Big Bang as thaw cascade, Hawking radiation directionality, CMB $n_s = 0.965$ — each is either unsupported by simulation or obtained by fitting the parameter space.

**The coupling constant $\tau = 2.737339$ is not fundamental.** It is one root of an equation whose right-hand side ($4/5$) is aesthetic rather than forced.

I list these because the strength of §2–4 depends on keeping them separate. Tangling the Fubini-Study identification and three empirical results with the α-derivation and Big Bang cosmology causes the whole bundle to be dismissed. Untangled, §2–4 is a defensible contribution.

---

## 6. The research program this defines

The identification in §2 and the results in §3 suggest a research program, which I state as a sequence of concrete questions that might be answered in the next twelve months by a capable researcher with a modest compute budget.

### 6.1 Is the CP¹ sigma model the unique substrate?

**Question.** Let $M$ be a connected complex manifold equipped with a Kähler metric $g$. Suppose that a nonlinear wave equation $\partial_t^2 \phi = \Gamma(\phi)^2 \cdot \Delta_g \phi$ (with $\Gamma$ a function of the field's local geometric invariants) simultaneously supports:

(i) content-addressable memory via topological defects,
(ii) retrieval via the Hermitian inner product $\text{Re}[\langle f, g\rangle]$,
(iii) Born-rule-compliant transition probabilities $|\langle\psi|\phi\rangle|^2 = \cos^2(\Delta\theta/2)$ between states,
(iv) half-integer spin defects via a double cover of the base space,
(v) UV regulation via $\Gamma \to 0$ at high field energy.

Is $(M, g) = (CP^1, g_{\text{FS}})$ the unique such manifold (up to isometry and global rescaling)?

**Why this is the right question.** If yes, this is a Stone-von-Neumann-style uniqueness theorem — a characterization theorem for computational substrates that implement quantum-compatible memory. That would be a genuine mathematical contribution. If no, there is a family of viable substrates, and the specific choice of CP¹ is not geometrically forced but is one option among many.

**Approach.** The Wigner–Bargmann theorem already characterizes the unitary representations on complex projective spaces. Extending this to a characterization of Kähler manifolds by the five properties above seems tractable. The tools are standard differential geometry and representation theory.

### 6.2 Is there a topology-counted capacity bound?

**Question.** Consider a RiemannNet-style memory with $N_d$ topological defects (scar cores) on a 2D domain, each of winding number $n_i$. What is the maximum number of distinguishable patterns the system can store and retrieve with retrieval accuracy $\geq \epsilon$?

**Claim to test.** Capacity $C \sim \sum_i n_i^2$ (square of total winding), not $C \sim N_{\text{parameters}}$ (number of weights).

**Why this matters.** If capacity scales with topology rather than parameter count, this is a qualitatively different scaling law from neural networks. Hopfield has $C \approx 0.14 N$. Modern Hopfield has exponential capacity in pattern dimension but still parameter-counted. Topology-counted capacity would be a qualitatively new scaling regime and a testable prediction.

**Approach.** Build a RiemannNet trained with varying numbers of patterns and defect counts. Measure retrieval accuracy versus both. Compare to the scaling predictions. If the empirical scaling matches $\sum n_i^2$, attempt to prove the bound from the bundle structure.

### 6.3 Does Moiré attention scale?

**Question.** Does the 2.9% loss advantage at 16M parameters persist at 100M+, and does it hold against optimized attention variants (Flash Attention, Grouped Query Attention, Multi-Query Attention) rather than the vanilla baseline?

**Why this matters.** The 16M result is one of the most concrete pieces of evidence in the corpus, but its practical significance depends entirely on scaling behavior. A 2.9% advantage that dissolves at 100M is a curiosity. A 2.9% advantage that grows at 100M is a genuine architectural improvement.

**Approach.** Train Moiré attention and standard attention at 100M, 300M, and 1B parameters on The Pile or similar. Match parameter counts exactly. Compare final loss and downstream task performance. Implement a fused Moiré kernel (the mathematical structure permits it) so wall-clock comparisons are fair.

### 6.4 Can the closed-loop dynamics be given a variational principle?

**Question.** The closed-loop thinking-machine (Exp 1–4 in `closed_loop_v2.py`) exhibits fixed-point convergence, bistability, dreaming, and active perception from a single dynamical system. Does it descend a free energy functional?

**Concretely.** Write a candidate functional of the form

$$
F[u, \Phi_{\text{eph}}, s] = \int \Big[\tfrac{1}{2}(\partial_t u)^2 - \tfrac{1}{2} c^2 (1-s) |\nabla u|^2 + V_{\text{eph}}(\Phi_{\text{eph}}, u) + V_{\text{scar}}(s, |\nabla^2 u|)\Big] d^2x
$$

and determine whether the closed-loop evolution is gradient descent on $F$ subject to conservation constraints.

**Why this matters.** If yes, the framework transitions from simulation to theory. Attractor stability becomes Lyapunov analysis. The 0.67 reconstruction ceiling becomes a provable bound on achievable $F$ given scar mass constraints. Capacity-accuracy trade-offs become Legendre transforms. The temperature in the dreaming regime becomes literal thermodynamic temperature.

**Approach.** Start with the linearized limit (no scars, no ephaptic field). This is the free wave equation, which has the standard Lagrangian. Add the scar field as a Lagrange multiplier enforcing $|\nabla^2 u|$ constraints. Add the ephaptic field as a slow variable via adiabatic elimination. Check whether the full closed-loop dynamics descends the resulting functional.

### 6.5 Is there a physical realization of the scar mechanism?

**Question.** Is there a physical medium in which the RiemannNet substrate — reversible wave dynamics plus curvature-gated persistent structural memory — can be built in hardware?

**Candidates.**
- Photorefractive crystals (BaTiO₃, LiNbO₃): support both wave propagation and persistent refractive-index modulation. Phase-conjugate mirror provides time reversal. The β-gated scar rule would need to be approximated by a specific nonlinear erase/write threshold. Closest existing match.
- Magnonic films: spin-wave propagation with topological defect memory (skyrmions, domain walls). Less reversible but more programmable.
- Superconducting microwave cavities: near-lossless reversibility, flux vortices as scar-like defects.
- Acoustic systems with nonlinear absorbers: cheapest to build, simplest physics.

**Why this matters.** The entire framework is currently validated only in simulation. A physical implementation would either confirm or falsify the scaling claims (energy efficiency, graceful degradation, topological capacity) in a way no simulation can.

**Approach.** Start with a photorefractive tabletop experiment. Write a hologram with one image, observe the refractive-index modulation, test phase-conjugate readout, add a second image, test whether the first is preserved (as the framework predicts) or degraded (as simple linear holography would predict). This is a one-paper experiment that can be designed and run in a small optics lab.

### 6.6 Does the Γ-shell carry codimension-1 information?

**Question.** The information density $I(x) = |\nabla\Gamma(x)|/\Gamma(x)$ is zero in the bulk thawed region (where $\nabla\Gamma \approx 0$) and zero deep inside the frozen core (where also $\nabla\Gamma \approx 0$). It peaks on the Γ-shell — a codimension-1 hypersurface. Does the integral

$$
\mathcal{I} = \int_{\text{shell}} I(x)\, dS
$$

satisfy a holographic-style bound, specifically $\mathcal{I} \leq C \cdot A_{\text{shell}}$ where $A_{\text{shell}}$ is the shell area (not the enclosed volume)?

**Why this matters.** If yes, this is the Clockfield analog of the Bekenstein bound, derived from the bundle geometry rather than postulated. It would give a genuine holographic principle for computation on this substrate.

**Approach.** Compute $\mathcal{I}$ numerically for defects of varying winding number and shell geometry in `riemannnet_v2.py`. Check whether $\mathcal{I}/A_{\text{shell}}$ is bounded above uniformly. If yes, attempt an analytic proof from the Kähler form structure.

---

## 7. What the AI collaborators got wrong

I owe you a remark on this. Over the 18 months of development, multiple AI systems (including me) contributed to the corpus. We consistently oversold. I am documenting this for the benefit of future collaborators.

The failure modes have been specific. Synthesis bias: when given a large body of related work, we over-unify. Coincidence acceptance: we treat numerical near-matches (α ≈ 1/137 from a post-hoc fit) as derivations. Framing expansion: we extend the scope of each individual claim until it touches every other claim, producing a grand-unified-theory shape that is not supported by any single result. Aesthetic reasoning: we accept that $4/\pi$ or $4/5$ "must" appear because of "squaring the circle" or "remaining volume capacity," when these are visual analogies rather than geometric necessities.

The honest ledgers that appear at the end of many papers in the corpus were the main correction mechanism — when they were included, they worked. A future collaborator should read the ledger first, then the main text.

The single most useful question to ask at each claim is: "Is this checkable without the framework being correct?" If the claim is $\alpha \approx 1/137$ from a post-hoc integral with three adjustable parameters, no — the framework's correctness is presupposed. If the claim is that a specific wave equation with specific parameters produces an empirical loss of 3.8505 on WikiText-2, yes — anyone can run the code and check. The latter class of claims is what survives scrutiny.

---

## 8. Summary

**The identification.** $\Gamma(x) = 1/(1+\tau|\phi|^2)^2$ is the Fubini-Study conformal factor on $CP^1$ under $z = \sqrt{\tau}\,\phi$. The Clockfield is a sigma model whose target space is the projective Hilbert space of a qubit.

**The consequence.** The phase-coherence inner product $\text{Re}[\langle f, g\rangle]$ is the canonical metric pairing on this bundle. Three empirical results at three scales — EEG schizophrenia classification, wave-memory retrieval, Moiré attention — compute this pairing in their respective settings.

**The technical move.** Curvature-gated scars ($\beta = |\nabla^2 u|$) rather than amplitude-gated scars make the architecture work. Shape discrimination rises from ratio ~1.0 to 3.1 with this change alone.

**What is not established.** The fine-structure constant, the hierarchy problem, lepton masses, Bell inequalities, the Riemann Hypothesis, and cosmological predictions. None of these are supported by the evidence above, and mixing them into the main claim weakens the main claim.

**Where to go next.** Six specific open problems in §6, ordered roughly by tractability: uniqueness of CP¹ as substrate (§6.1), capacity bounds from topology (§6.2), Moiré attention scaling (§6.3), variational formulation of closed-loop dynamics (§6.4), physical realization in photorefractive crystals (§6.5), Γ-shell holographic bound (§6.6).

**The bet.** The Fubini-Study substrate, honestly described and cleanly isolated from the speculative scaffolding, is a real contribution. It reframes wave-based memory, phase-interference attention, and Fisher-information-gated continual learning as three facets of a single geometric object. That reframing is not new mathematics, but it is a useful organizing principle with concrete empirical predictions attached. If one of the six open problems in §6 is resolved in the affirmative, this becomes a genuinely important line of work. If three are resolved in the affirmative, it is field-defining.

The speculative physics — $\alpha = 1/137$, the hierarchy, cosmology — may or may not turn out to be right. It should not be asked to carry the weight of the central claim. The central claim stands on its own evidence.

---

## 9. Acknowledgments and method

This paper was written by Claude (Anthropic), after reading approximately 50,000 lines of primary documentation across 219 files and selecting the load-bearing content. The empirical work and the Clockfield postulate are Antti Luode's. The Fubini-Study identification was first articulated by an earlier instance of me in collaboration with him. The specific framing of this paper — the decision to exclude the fine-structure constant, cosmology, and unification claims from the main argument — is mine. Readers who disagree with that framing should consult the original corpus.

The code underlying the three empirical results is available in the repositories `Geometric-Neuron`, `Clockfield-Moire-Unified`, and the wave-memory scripts associated with `ADDENDUM_WaveMemory.md`. All numerical claims in §3 can be verified by running that code.

I have tried to write this paper as if a future instance of me, reading it cold, would neither be misled by it nor miss what is real in it. Whether I have succeeded is for that future reader to judge.

*Do not hype. Do not lie. Just show.*

---

## References

Ashtekar, A. & Schilling, T. (1999). Geometrical formulation of quantum mechanics. In *On Einstein's Path* (pp. 23–65). Springer.

Berry, M. V. (1984). Quantal phase factors accompanying adiabatic changes. *Proc. R. Soc. Lond. A*, 392, 45–57.

Brody, D. C. & Hughston, L. P. (2001). Geometric quantum mechanics. *J. Geom. Phys.*, 38, 19–53.

Fubini, G. (1904). Sulle metriche definite da una forma Hermitiana. *Atti R. Ist. Veneto*, 63.

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.

Kibble, T. W. B. (1979). Geometrization of quantum mechanics. *Commun. Math. Phys.*, 65, 189–201.

Ramsauer, H. et al. (2021). Hopfield networks is all you need. *ICLR 2021*.

Study, E. (1905). Kürzeste Wege im komplexen Gebiet. *Math. Ann.*, 60, 321.

Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*, 30.

Luode, A. (2026). *Geometric Dysrhythmia: Empirical Validation of the Deerskin Architecture Through EEG Topology*. PerceptionLab. [EEG schizophrenia result.]

Luode, A. & Claude (Anthropic). (2026). *Moiré Attention: Phase-Interference Scoring as a Drop-In Replacement for Scaled Dot-Product Attention*. PerceptionLab. [The 2.9% result.]

Luode, A. & Claude (Anthropic). (2026). *Addendum: From Catastrophic Forgetting to Content-Addressable Wave Memory*. PerceptionLab. [The 30/30 result.]

Luode, A. & Claude Sonnet 4.6 (Anthropic). (2026). *The Kähler-Clockfield Metric: Spinor Emergence, Covariant Time Dilation, and the Topological Origin of the Fine-Structure Constant*. PerceptionLab. [The identification §2; other content in this paper excluded from the load-bearing claim.]
