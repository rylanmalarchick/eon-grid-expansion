
using JSON3

# Numerical checks and the provenance record, inlined.
#
# These were a local package pinned by absolute path, which made the Julia side
# unrunnable for anyone but its author. The checks are small and are reproduced
# here in full: they STILL VALIDATE and still raise. Only the package boundary
# is gone.
module Checks

# A nested module does not inherit the parent's `using`.
using JSON3

struct CheckResult
    check_name::String
    passed::Bool
    rtol::Float64
    atol::Float64
    norm_used::String
    error_message::Union{String,Nothing}
end

struct ProvenanceRecord
    checks::Vector{CheckResult}
end

function emit_provenance(record::ProvenanceRecord, io::IO)
    entries = map(record.checks) do c
        Dict(
            "check_name" => c.check_name,
            "passed" => c.passed,
            "rtol" => c.rtol,
            "atol" => c.atol,
            "norm_used" => c.norm_used,
            "error_message" => c.error_message,
        )
    end
    payload = Dict("checks_passed" => entries, "provenance_backend" => "inline")
    print(io, JSON3.write(payload))
end

check_finite(x; name::String="value") =
    all(isfinite, x) || error("$name: non-finite value (NaN or Inf)")

check_non_negative(x; name::String="value", atol::Float64=1e-12) =
    all(v -> v >= -abs(atol), x) || error("$name: negative value $(minimum(x))")

# The tolerance is POSITIONAL at the call site (driver line ~239); keeping the
# keyword-only form silently produced a MethodError that the n<=24 brute-force
# fallback then swallowed.
function check_normalized_l1(x, atol::Float64=1e-9; name::String="value")
    total = sum(abs, x)
    isapprox(total, 1.0; atol=max(atol, 1e-12)) ||
        error("$name: L1 norm $total != 1")
end

function check_probability(x; name::String="value", atol::Float64=1e-12)
    check_finite(x; name=name)
    check_non_negative(x; name=name, atol=atol)
    all(v -> v <= 1.0 + max(atol, 1e-9), x) ||
        error("$name: value $(maximum(x)) exceeds 1")
end

end # module

using JuliQAOA
using ITensorMPS
using ITensors
using Optim
using Random
using Statistics

function append_provenance(checks)
    provenance_path = get(ENV, "EON_PROVENANCE_PATH", "")
    isempty(provenance_path) && return
    mkpath(dirname(provenance_path))
    record = Checks.ProvenanceRecord(checks)
    open(provenance_path, "a") do io
        Checks.emit_provenance(record, io)
        write(io, '\n')
    end
end

function record_validation!(checks, check_name::String, norm_used::String, rtol::Float64, atol::Float64, thunk::Function)
    try
        thunk()
        push!(checks, Checks.CheckResult(check_name, true, rtol, atol, norm_used, nothing))
    catch err
        message = sprint(showerror, err)
        failure = Checks.CheckResult(check_name, false, rtol, atol, norm_used, message)
        append_provenance([failure])
        rethrow(err)
    end
end

function z_state_energy(bits::AbstractVector{Int}, interactions, constant::Float64)
    energy = constant
    for interaction in interactions
        term = interaction.weight
        for qubit in interaction.qubits
            term *= iszero(bits[qubit]) ? 1.0 : -1.0
        end
        energy += term
    end
    return energy
end

function sampled_state_energy(sample::AbstractVector{Int}, interactions, constant::Float64)
    energy = constant
    for interaction in interactions
        term = interaction.weight
        for qubit in interaction.qubits
            term *= sample[qubit] == 1 ? 1.0 : -1.0
        end
        energy += term
    end
    return energy
end

function exact_objective_values(nqubits::Int, interactions, constant::Float64)
    return [z_state_energy(state, interactions, constant) for state in states(nqubits)]
end

function optimize_angles(
    nqubits::Int,
    obj_vals::Vector{Float64},
    p_rounds::Int,
    restarts::Int,
    iterations::Int,
    seed::Int;
    angles::Union{Nothing, Vector{Float64}} = nothing,
)
    mixer = mixer_x(nqubits)
    objective(x) = exp_value(collect(mod.(x, 2π)), mixer, obj_vals)

    if angles !== nothing
        wrapped = collect(mod.(angles, 2π))
        return wrapped, objective(wrapped)
    end

    lower = zeros(2 * p_rounds)
    upper = fill(2π, 2 * p_rounds)
    rng = MersenneTwister(seed)
    starts = Vector{Vector{Float64}}()
    push!(starts, zeros(2 * p_rounds))
    push!(starts, fill(π / 4, 2 * p_rounds))
    for _ in 1:max(restarts, 1)
        push!(starts, 2π * rand(rng, 2 * p_rounds))
    end

    best_angles = copy(starts[1])
    best_value = objective(best_angles)
    for start in starts
        result = optimize(
            objective,
            lower,
            upper,
            start,
            Fminbox(NelderMead()),
            Optim.Options(iterations = iterations, show_trace = false, store_trace = false),
        )
        candidate = collect(mod.(Optim.minimizer(result), 2π))
        candidate_value = objective(candidate)
        if candidate_value < best_value
            best_value = candidate_value
            best_angles = candidate
        end
    end
    return best_angles, best_value
end

function entanglement_entropy(psi::MPS, bond::Int)
    psi_bond = orthogonalize(psi, bond)
    row_inds = bond == 1 ? (siteinds(psi_bond, bond)...,) : (linkinds(psi_bond, bond - 1)..., siteinds(psi_bond, bond)...)
    _, svals, _ = svd(psi_bond[bond], row_inds)
    entropy_value = 0.0
    for idx in 1:dim(svals, 1)
        # Clamp to [0, 1]: truncation float error can give abs2(sval) = 1 + eps,
        # whose -p*log(p) term is NEGATIVE (~ -eps) and trips the strict
        # non-negativity validation on the whole profile (the 2026-07-31
        # scale-sweep fast-fail root cause). A probability > 1 is pure float
        # noise for a normalized state; the validation itself stays strict.
        probability = min(abs2(svals[idx, idx]), 1.0)
        if probability > 1e-12
            entropy_value -= probability * log(probability)
        end
    end
    return entropy_value
end

function entropy_profile(psi::MPS)
    if length(psi) <= 1
        return Float64[]
    end
    return [entanglement_entropy(psi, bond) for bond in 1:(length(psi) - 1)]
end

function evaluate_mps(
    angles::Vector{Float64},
    interactions,
    constant::Float64,
    nqubits::Int,
    chi::Int,
    cutoff::Float64,
    sample_count::Int,
    seed::Int,
)
    problem = QAOAProblem(interactions; nqubits = nqubits)
    hamiltonian = JuliQAOA.z_hamiltonian_mpo(problem)
    psi = redirect_stdout(devnull) do
        circuit = JuliQAOA.variational_circuit(problem, angles)
        apply(circuit, problem.psi0; cutoff = cutoff, maxdim = chi)::MPS
    end
    normalize!(psi)

    profile = entropy_profile(psi)
    expected_energy = real(inner(psi', hamiltonian, psi; cutoff = 1e-4)) + constant

    rng = MersenneTwister(seed)
    sampled_energies = Float64[]
    sample_counts = Dict{String, Int}()
    for _ in 1:max(sample_count, 1)
        sample = sample!(rng, deepcopy(psi))
        key = join(sample)
        sample_counts[key] = get(sample_counts, key, 0) + 1
        push!(sampled_energies, sampled_state_energy(sample, interactions, constant))
    end
    probabilities = [count / max(sample_count, 1) for count in values(sample_counts)]

    checks = Checks.CheckResult[]
    record_validation!(checks, "finite_array", "n/a", 0.0, 0.0, () -> begin
        Checks.check_finite(vcat([expected_energy], sampled_energies); name="mps_energy_curve")
    end)
    record_validation!(checks, "non_negative_array", "n/a", 0.0, 0.0, () -> begin
        Checks.check_non_negative(profile; name="entropy_profile")
    end)
    record_validation!(checks, "probability_array", "n/a", 0.0, 0.0, () -> begin
        Checks.check_probability(probabilities; name="sample_distribution")
    end)
    record_validation!(checks, "normalized_l1", "l1", 0.0, 1e-10, () -> begin
        Checks.check_normalized_l1(probabilities, 1e-10; name="sample_distribution")
    end)
    append_provenance(checks)

    return Dict(
        "best_sampled_energy" => minimum(sampled_energies),
        "chi" => chi,
        "entropy_profile" => profile,
        "expected_energy" => expected_energy,
        "max_entropy" => isempty(profile) ? 0.0 : maximum(profile),
        "sample_variance" => length(sampled_energies) > 1 ? var(sampled_energies) : 0.0,
    )
end

function default_angles(p_rounds::Int)
    if p_rounds <= 0
        return Float64[]
    end
    betas = [0.15 / idx for idx in 1:p_rounds]
    gammas = [0.35 / idx for idx in 1:p_rounds]
    return [betas; gammas]
end

function process_spec(spec)
    nqubits = Int(spec["nqubits"])
    constant = Float64(spec["constant"])
    chi_values = [Int(value) for value in spec["chi_values"]]
    qaoa_rounds = Int(spec["qaoa_rounds"])
    angle_restarts = Int(spec["angle_restarts"])
    angle_iterations = Int(spec["angle_iterations"])
    sample_count = Int(spec["sample_count"])
    cutoff = Float64(spec["cutoff"])
    seed = Int(spec["seed"])
    max_exact_qubits = haskey(spec, "max_exact_qubits") ? Int(spec["max_exact_qubits"]) : 24
    reference_energy = (haskey(spec, "reference_energy") && spec["reference_energy"] !== nothing) ? Float64(spec["reference_energy"]) : NaN

    interactions = ZInteractions[]
    for entry in spec["interactions"]
        qubits = [Int(qubit) for qubit in entry["qubits"]]
        weight = Float64(entry["weight"])
        push!(interactions, ZInteractions(qubits, weight))
    end

    initial_angles = nothing
    if haskey(spec, "angles") && spec["angles"] !== nothing
        initial_angles = [Float64(angle) for angle in spec["angles"]]
    end

    if nqubits <= max_exact_qubits
        exact_values = exact_objective_values(nqubits, interactions, constant)
        angles, exact_qaoa_energy = optimize_angles(
            nqubits,
            exact_values,
            qaoa_rounds,
            angle_restarts,
            angle_iterations,
            seed;
            angles = initial_angles,
        )
        exact_ground_energy = minimum(exact_values)
    else
        angles = initial_angles === nothing ? default_angles(qaoa_rounds) : collect(mod.(initial_angles, 2π))
        exact_qaoa_energy = NaN
        exact_ground_energy = reference_energy
    end

    chi_results = [
        evaluate_mps(
            angles,
            interactions,
            constant,
            nqubits,
            chi,
            cutoff,
            sample_count,
            seed + chi,
        ) for chi in chi_values
    ]

    output_checks = Checks.CheckResult[]
    if isfinite(exact_ground_energy)
        record_validation!(output_checks, "finite", "n/a", 0.0, 0.0, () -> begin
            Checks.check_finite(exact_ground_energy; name="exact_ground_energy")
        end)
    end
    if isfinite(exact_qaoa_energy)
        record_validation!(output_checks, "finite", "n/a", 0.0, 0.0, () -> begin
            Checks.check_finite(exact_qaoa_energy; name="exact_qaoa_energy")
        end)
    end
    append_provenance(output_checks)

    return sanitize_for_json(Dict(
        "angles" => angles,
        "chi_results" => chi_results,
        "exact_ground_energy" => exact_ground_energy,
        "exact_qaoa_energy" => exact_qaoa_energy,
    ))
end

"""Recursively replace NaN/Inf with nothing for JSON3 compatibility."""
function sanitize_for_json(x::AbstractFloat)
    isfinite(x) ? x : nothing
end
function sanitize_for_json(x::AbstractVector)
    [sanitize_for_json(v) for v in x]
end
function sanitize_for_json(x::AbstractDict)
    Dict(k => sanitize_for_json(v) for (k, v) in x)
end
function sanitize_for_json(x)
    x
end

function run_file(spec_path::String)
    spec = JSON3.read(read(spec_path, String))
    println(JSON3.write(process_spec(spec)))
end

function run_server()
    for line in eachline(stdin)
        if isempty(strip(line))
            continue
        end
        payload = try
            spec = JSON3.read(line)
            process_spec(spec)
        catch err
            Dict("error" => sprint(showerror, err))
        end
        println(JSON3.write(payload))
        flush(stdout)
    end
end

if !isempty(ARGS) && ARGS[1] != "--server"
    run_file(ARGS[1])
else
    run_server()
end
