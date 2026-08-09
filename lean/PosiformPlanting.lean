/-
Machine-checked core of the posiform-planting argument (arXiv:2308.05859)
used by src/eon/instances/external.py.

What IS proven here (core Lean 4, no mathlib):
  1. term_nonneg          -- each posiform term b * z * z' (b >= 0, binary
                             factors) is nonnegative;
  2. zero_achiever_is_global_min
                          -- an assignment zeroing every term is a global
                             minimizer of the posiform sum;
  3. min_attainer_satisfies_all_clauses
                          -- any assignment attaining the value 0 zeroes every
                             term, i.e. satisfies every 2-SAT clause.

What is NOT proven here: uniqueness of the planted optimum. By (3), a second
optimum would be a second 2-SAT solution; the generator excludes that per
instance with an exhaustive 2-SAT uniqueness check (implication-graph SCC +
per-variable forced-flip UNSAT), and tests brute-force it at small n. That
part is machine-checked-numeric per instance, not a theorem.
-/

namespace PosiformPlanting

/-- Sum of a list of nonnegative integers is nonnegative. -/
theorem sum_nonneg : ∀ (l : List Int), (∀ v ∈ l, 0 ≤ v) → 0 ≤ l.sum
  | [], _ => Int.le_refl 0
  | v :: rest, h => by
      have hv : 0 ≤ v := h v (List.mem_cons_self ..)
      have hr : 0 ≤ rest.sum :=
        sum_nonneg rest (fun w hw => h w (List.mem_cons_of_mem _ hw))
      simpa [List.sum_cons] using Int.add_nonneg hv hr

/-- A sum of nonnegative integers is zero only if every summand is zero. -/
theorem eq_zero_of_sum_eq_zero :
    ∀ (l : List Int), (∀ v ∈ l, 0 ≤ v) → l.sum = 0 → ∀ v ∈ l, v = 0
  | [], _, _, v, hv => absurd hv (List.not_mem_nil)
  | w :: rest, hnn, hsum, v, hv => by
      have hw : 0 ≤ w := hnn w (List.mem_cons_self ..)
      have hr : 0 ≤ rest.sum :=
        sum_nonneg rest (fun u hu => hnn u (List.mem_cons_of_mem _ hu))
      have hsum' : w + rest.sum = 0 := by simpa [List.sum_cons] using hsum
      rcases List.mem_cons.mp hv with h | h
      · omega
      · exact eq_zero_of_sum_eq_zero rest
          (fun u hu => hnn u (List.mem_cons_of_mem _ hu)) (by omega) v h

/-- A posiform term b * z * z' with b >= 0 and binary factors is nonnegative. -/
theorem term_nonneg (b z z' : Int) (hb : 0 ≤ b)
    (hz : z = 0 ∨ z = 1) (hz' : z' = 0 ∨ z' = 1) : 0 ≤ b * z * z' := by
  rcases hz with rfl | rfl <;> rcases hz' with rfl | rfl <;> simp [hb]

/-- An assignment zeroing every (nonnegative) term is a global minimizer of the
posiform sum: its value is 0 and every other value is >= 0. -/
theorem zero_achiever_is_global_min {α : Type} (terms : List (α → Int))
    (nonneg : ∀ t ∈ terms, ∀ y, 0 ≤ t y)
    (xstar : α) (vanish : ∀ t ∈ terms, t xstar = 0) (x : α) :
    (terms.map (fun t => t xstar)).sum = 0 ∧
      (terms.map (fun t => t xstar)).sum ≤ (terms.map (fun t => t x)).sum := by
  have hzero : (terms.map (fun t => t xstar)).sum = 0 := by
    have hall : ∀ v ∈ terms.map (fun t => t xstar), v = 0 := by
      intro v hv
      obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hv
      exact vanish t ht
    -- A list of zeros sums to zero.
    have : ∀ (l : List Int), (∀ v ∈ l, v = 0) → l.sum = 0 := by
      intro l
      induction l with
      | nil => intro _; rfl
      | cons w rest ih =>
          intro h
          have hw : w = 0 := h w (List.mem_cons_self ..)
          have hrest : rest.sum = 0 :=
            ih (fun u hu => h u (List.mem_cons_of_mem _ hu))
          simp [List.sum_cons, hw, hrest]
    exact this _ hall
  refine ⟨hzero, ?_⟩
  have hnn : 0 ≤ (terms.map (fun t => t x)).sum := by
    apply sum_nonneg
    intro v hv
    obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hv
    exact nonneg t ht x
  omega

/-- Any assignment attaining the value 0 zeroes every term -- i.e. satisfies
every 2-SAT clause. Uniqueness of the QUBO optimum therefore reduces to
uniqueness of the 2-SAT solution (checked per instance by the generator). -/
theorem min_attainer_satisfies_all_clauses {α : Type} (terms : List (α → Int))
    (nonneg : ∀ t ∈ terms, ∀ y, 0 ≤ t y) (x : α)
    (hzero : (terms.map (fun t => t x)).sum = 0) :
    ∀ t ∈ terms, t x = 0 := by
  intro t ht
  have hmem : t x ∈ terms.map (fun t => t x) :=
    List.mem_map.mpr ⟨t, ht, rfl⟩
  have hnn : ∀ v ∈ terms.map (fun t => t x), 0 ≤ v := by
    intro v hv
    obtain ⟨u, hu, rfl⟩ := List.mem_map.mp hv
    exact nonneg u hu x
  exact eq_zero_of_sum_eq_zero _ hnn hzero (t x) hmem

end PosiformPlanting

/-
Axiom audit. Without this the documented check (`lean lean/PosiformPlanting.lean`)
prints NOTHING and exits 0, which is indistinguishable from a file that proves
nothing at all. Each theorem below should report only the standard axioms
(propext, Classical.choice, Quot.sound) with no `sorry` and no bespoke axiom.

SCOPE, stated here so it cannot be read off the filename: these theorems are
about a posiform whose terms are pointwise NONNEGATIVE, which is
`generate_posiform_planted`. They do NOT cover `generate_fused_planted`, whose
blocks draw coefficients from (-10, 10) and is the generator the experiments
actually use. They also cover the un-expanded posiform, not the four-branch
sign expansion.
-/
#print axioms PosiformPlanting.sum_nonneg
#print axioms PosiformPlanting.eq_zero_of_sum_eq_zero
#print axioms PosiformPlanting.term_nonneg
#print axioms PosiformPlanting.zero_achiever_is_global_min
#print axioms PosiformPlanting.min_attainer_satisfies_all_clauses
