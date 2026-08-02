# Exact tensor-network control for the hardness classifier.
#
# Contracts the Ising/QUBO as a tensor network via GenericTensorNetworks.jl:
#   - TreeSA finds a near-optimal contraction order; its space complexity `sc` is
#     the contraction width = log2 of the largest intermediate tensor = the bond
#     dimension 2^sc an exact tensor network needs (reading.txt R30/R18). `tc` is
#     the time complexity (log2 of the contraction FLOPs).
#   - When sc and tc are both within budget the network is contracted exactly
#     (tropical / min-plus semiring) for the EXACT ground energy -- ground truth
#     the MPS sweep is scored against, and a rigorous TN-easy certificate.
#   - A width above the chi budget is a HEURISTIC hardness signal only: TreeSA
#     returns an upper bound on the optimal contraction width, so a large sc cannot
#     certify that no narrow contraction exists. Labelled as heuristic, never a
#     certificate.
#
# Same JSON-in/JSON-out contract and provenance discipline as juliqaoa_driver.jl.

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

using GenericTensorNetworks
using Graphs
using Random

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

function record_validation!(checks, check_name::String, thunk::Function)
    try
        thunk()
        push!(checks, Checks.CheckResult(check_name, true, 0.0, 0.0, "n/a", nothing))
    catch err
        message = sprint(showerror, err)
        push!(checks, Checks.CheckResult(check_name, false, 0.0, 0.0, "n/a", message))
        append_provenance(checks)
        rethrow(err)
    end
end

"""Build a SpinGlass: 1-body terms -> fields h, 2-body terms -> edge weights J.

J must be ordered to match `edges(g)` iteration, not insertion order, so weights
are looked up from a dict keyed by the sorted vertex pair."""
function build_spinglass(nqubits::Int, interactions)
    h = zeros(Float64, nqubits)
    edgeweight = Dict{Tuple{Int,Int}, Float64}()
    for entry in interactions
        qubits = [Int(q) for q in entry["qubits"]]
        weight = Float64(entry["weight"])
        if length(qubits) == 1
            h[qubits[1]] += weight
        elseif length(qubits) == 2
            a, b = minmax(qubits[1], qubits[2])
            a == b && error("self-coupling on qubit $a is not a valid 2-body term")
            edgeweight[(a, b)] = get(edgeweight, (a, b), 0.0) + weight
        else
            error("GTN control supports at most 2-body terms; got $(length(qubits))-body")
        end
    end
    g = SimpleGraph(nqubits)
    for (a, b) in keys(edgeweight)
        add_edge!(g, a, b)
    end
    J = [edgeweight[(min(src(e), dst(e)), max(src(e), dst(e)))] for e in edges(g)]
    return g, J, h
end

function process_spec(spec)
    nqubits = Int(spec["nqubits"])
    constant = Float64(spec["constant"])
    memory_sc_budget =
        haskey(spec, "memory_sc_budget") ? Float64(spec["memory_sc_budget"]) : 28.0
    time_tc_budget =
        haskey(spec, "time_tc_budget") ? Float64(spec["time_tc_budget"]) : 38.0
    seed = haskey(spec, "seed") ? Int(spec["seed"]) : 0
    treesa_ntrials = haskey(spec, "treesa_ntrials") ? Int(spec["treesa_ntrials"]) : 10
    treesa_niters = haskey(spec, "treesa_niters") ? Int(spec["treesa_niters"]) : 50

    interactions = spec["interactions"]

    # Trivial instance: no spins -> energy is the constant offset only.
    if nqubits <= 0
        return Dict(
            "backend" => "generic_tn_tropical",
            "contraction_width" => 0.0,
            "time_complexity" => 0.0,
            "exact_ground_energy" => constant,
            "within_budget" => true,
            "chi_required" => 1.0,
            "nqubits" => 0,
        )
    end

    g, J, h = build_spinglass(nqubits, interactions)
    problem = SpinGlass(g, J, h)

    # TreeSA is stochastic; seed the global RNG and use enough trials that the
    # reported width is a stable near-optimal upper bound. Residual run-to-run
    # variation near the threshold is handled by the not-mps_easy guard on the
    # Python side, not by exact reproducibility here.
    Random.seed!(seed)
    net = GenericTensorNetwork(
        problem; optimizer = TreeSA(ntrials = treesa_ntrials, niters = treesa_niters)
    )
    cc = contraction_complexity(net)
    sc = Float64(cc.sc)
    tc = Float64(cc.tc)

    # Contract exactly only when BOTH the memory (sc) and time (tc) budgets hold:
    # solve cost ~ 2^tc, so an sc-cheap but tc-expensive instance must not be
    # contracted (it would otherwise run to the subprocess timeout).
    within_budget = sc <= memory_sc_budget && tc <= time_tc_budget
    exact_ground = nothing
    if within_budget
        exact_ground = Float64(solve(net, SizeMin())[].n) + constant
    end

    checks = Checks.CheckResult[]
    record_validation!(checks, "finite", () -> Checks.check_finite(sc; name = "contraction_width"))
    if exact_ground !== nothing
        record_validation!(
            checks, "finite", () -> Checks.check_finite(exact_ground; name = "exact_ground_energy")
        )
    end
    append_provenance(checks)

    return Dict(
        "backend" => "generic_tn_tropical",
        "contraction_width" => sc,
        "time_complexity" => tc,
        "exact_ground_energy" => exact_ground,
        "within_budget" => within_budget,
        "chi_required" => within_budget ? 2.0^ceil(sc) : nothing,
        "nqubits" => nqubits,
    )
end

sanitize_for_json(x::AbstractFloat) = isfinite(x) ? x : nothing
sanitize_for_json(x::AbstractVector) = [sanitize_for_json(v) for v in x]
sanitize_for_json(x::AbstractDict) = Dict(k => sanitize_for_json(v) for (k, v) in x)
sanitize_for_json(x) = x

function run_file(spec_path::String)
    spec = JSON3.read(read(spec_path, String))
    println(JSON3.write(sanitize_for_json(process_spec(spec))))
end

function run_server()
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        payload = try
            sanitize_for_json(process_spec(JSON3.read(line)))
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
