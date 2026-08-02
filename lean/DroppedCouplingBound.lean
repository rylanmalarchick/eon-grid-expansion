/-
Machine-checked core of the D14 certificate's lower bound, implemented in
src/eon/quantum/decomposition.py (`dropped_coupling_lower_bound`).

The certificate pairs this proven lower bound against an achieved upper bound.
If the bound is wrong the certificate is worthless -- and worse, it is worthless
in the direction that flatters us, since a too-high "lower bound" makes the gap
look tighter than it is. It was previously justified by a paragraph of prose in
a docstring. This file replaces the prose with a proof.

What IS proven here (core Lean 4, no mathlib):
  1. sum_map_le_sum_map     -- a pointwise <= between two summands lifts to
                               their sums;
  2. dropped_term_ge_floor  -- a dropped coupling term J * z * z' with binary
                               factors is at least min 0 J (the ONLY place the
                               sign of J matters);
  3. decomposition_lower_bound
                            -- if every block contribution is at least its
                               claimed block minimum and every dropped term is
                               at least its floor, then offset + sum of block
                               minima + sum of floors is at most the energy of
                               EVERY assignment, i.e. it is a valid lower bound
                               on the global minimum;
  4. dropped_coupling_lower_bound
                            -- (3) instantiated at the floor of (2), which is
                               the bound the implementation computes.

Why the hypotheses are discharged in the implementation:
  * hBlockMin: each block minimum is computed by exhaustive enumeration over
    the block's binary assignments with the cardinality constraint RELAXED.
    Relaxing a constraint can only lower a minimum, so the enumerated value is
    at most the block's contribution under any feasible assignment.
  * hBinary: the surrogate's toggle variables are binary by construction.

SCOPE: the arithmetic is over Int. The argument uses only ordered-ring
reasoning -- no divisibility, no induction on magnitude -- so it transfers
verbatim to rational or real coefficients. The machine-checked statement is
nonetheless the Int one, and is labelled as such wherever it is cited.
-/

namespace DroppedCouplingBound

/-- A binary (0/1) integer, the shape of a surrogate toggle variable. -/
def IsBinary (z : Int) : Prop := z = 0 ∨ z = 1

/-- A pointwise `<=` between two functions lifts to a `<=` between the sums of
their images over a common list. This is the only structural lemma needed: the
bound is a sum of independently-bounded pieces. -/
theorem sum_map_le_sum_map {β : Type} (f g : β → Int) :
    ∀ (l : List β), (∀ b ∈ l, f b ≤ g b) →
      (l.map f).sum ≤ (l.map g).sum
  | [], _ => Int.le_refl 0
  | b :: rest, h => by
      have hb : f b ≤ g b := h b (List.mem_cons_self ..)
      have hr : (rest.map f).sum ≤ (rest.map g).sum :=
        sum_map_le_sum_map f g rest (fun c hc => h c (List.mem_cons_of_mem _ hc))
      simpa [List.map_cons, List.sum_cons] using Int.add_le_add hb hr

/-- A dropped coupling term `J * z * z'` with binary factors is at least
`min 0 J`. When `J >= 0` the term is nonnegative and the floor is 0; when
`J < 0` the worst case is both factors set, giving exactly `J`. This is the
only step where the sign of the coefficient is used. -/
theorem dropped_term_ge_floor (J z z' : Int)
    (hz : IsBinary z) (hz' : IsBinary z') :
    min 0 J ≤ J * z * z' := by
  rcases hz with rfl | rfl <;> rcases hz' with rfl | rfl <;>
    simp [Int.min_def] <;> split <;> omega

/-- The certificate's soundness, stated over an arbitrary assignment type.

`blocks` are the per-block energy contributions and `dropped` the inter-block
terms the decomposition discards. Given a floor for each, their sum plus the
offset is a lower bound on the energy of every assignment -- hence on the
global minimum, which is the energy of some assignment. -/
theorem decomposition_lower_bound {α : Type}
    (offset : Int)
    (blocks : List (α → Int)) (blockMin : (α → Int) → Int)
    (hBlockMin : ∀ b ∈ blocks, ∀ x, blockMin b ≤ b x)
    (dropped : List (α → Int)) (floor : (α → Int) → Int)
    (hFloor : ∀ d ∈ dropped, ∀ x, floor d ≤ d x)
    (x : α) :
    offset + (blocks.map blockMin).sum + (dropped.map floor).sum
      ≤ offset + (blocks.map (fun b => b x)).sum
          + (dropped.map (fun d => d x)).sum := by
  have hb : (blocks.map blockMin).sum ≤ (blocks.map (fun b => b x)).sum :=
    sum_map_le_sum_map _ _ blocks (fun b hb => hBlockMin b hb x)
  have hd : (dropped.map floor).sum ≤ (dropped.map (fun d => d x)).sum :=
    sum_map_le_sum_map _ _ dropped (fun d hd => hFloor d hd x)
  exact Int.add_le_add (Int.add_le_add (Int.le_refl offset) hb) hd

/-- The bound the implementation actually computes: block minima plus
`min 0 J` per dropped coupling. -/
theorem dropped_coupling_lower_bound {α : Type}
    (offset : Int)
    (blocks : List (α → Int)) (blockMin : (α → Int) → Int)
    (hBlockMin : ∀ b ∈ blocks, ∀ x, blockMin b ≤ b x)
    (coupling : List (Int × (α → Int) × (α → Int)))
    (hBinary : ∀ c ∈ coupling, ∀ x, IsBinary (c.2.1 x) ∧ IsBinary (c.2.2 x))
    (x : α) :
    offset + (blocks.map blockMin).sum
        + (coupling.map (fun c => min 0 c.1)).sum
      ≤ offset + (blocks.map (fun b => b x)).sum
          + (coupling.map (fun c => c.1 * c.2.1 x * c.2.2 x)).sum := by
  have hb : (blocks.map blockMin).sum ≤ (blocks.map (fun b => b x)).sum :=
    sum_map_le_sum_map _ _ blocks (fun b hb => hBlockMin b hb x)
  have hd : (coupling.map (fun c => min 0 c.1)).sum
      ≤ (coupling.map (fun c => c.1 * c.2.1 x * c.2.2 x)).sum := by
    refine sum_map_le_sum_map _ _ coupling (fun c hc => ?_)
    obtain ⟨h1, h2⟩ := hBinary c hc x
    exact dropped_term_ge_floor c.1 (c.2.1 x) (c.2.2 x) h1 h2
  exact Int.add_le_add (Int.add_le_add (Int.le_refl offset) hb) hd

end DroppedCouplingBound

/-
Axiom audit. Each theorem must rest only on Lean's standard axioms
(propext, Classical.choice, Quot.sound) with no `sorry` and no bespoke axiom.
-/
#print axioms DroppedCouplingBound.sum_map_le_sum_map
#print axioms DroppedCouplingBound.dropped_term_ge_floor
#print axioms DroppedCouplingBound.decomposition_lower_bound
#print axioms DroppedCouplingBound.dropped_coupling_lower_bound
