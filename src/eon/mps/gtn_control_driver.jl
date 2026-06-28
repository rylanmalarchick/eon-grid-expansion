# Exact tensor-network control for the hardness classifier.
#
# Replaces the degenerate treewidth-DP control (which faked a flat chi-curve by
# copying one exact energy to every chi). This driver contracts the Ising/QUBO as
# a real tensor network via GenericTensorNetworks.jl:
#   - TreeSA finds a near-optimal contraction order; its space complexity `sc` is
#     the contraction width = log2 of the largest intermediate tensor = the bond
#     dimension 2^sc an exact tensor network needs (reading.txt R30/R18).
#   - When sc is within the memory budget we contract exactly (tropical / min-plus
#     semiring) for the EXACT ground energy -- ground truth the MPS sweep is scored
#     against. This RIGOROUSLY certifies TN-easy (we solved it).
#   - sc above the chi budget is a HEURISTIC hardness signal only: TreeSA returns an
#     upper bound on the optimal contraction width, so a large sc cannot certify
#     that no narrow contraction exists. Labelled as heuristic, never a certificate.
#
# Same JSON-in/JSON-out contract and provenance discipline as juliqaoa_driver.jl.

using AgentBible
using JSON3
using GenericTensorNetworks
using Graphs
using Random

function append_provenance(checks)
    provenance_path = get(ENV, "EON_PROVENANCE_PATH", "")
    isempty(provenance_path) && return
    mkpath(dirname(provenance_path))
    record = AgentBible.ProvenanceRecord(checks)
    open(provenance_path, "a") do io
        AgentBible.emit_provenance(record, io)
        write(io, '\n')
    end
end

function record_validation!(checks, check_name::String, thunk::Function)
    try
        thunk()
        push!(checks, AgentBible.CheckResult(check_name, true, 0.0, 0.0, "n/a", nothing))
    catch err
        message = sprint(showerror, err)
        push!(checks, AgentBible.CheckResult(check_name, false, 0.0, 0.0, "n/a", message))
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
    seed = haskey(spec, "seed") ? Int(spec["seed"]) : 0

    interactions = spec["interactions"]

    # Trivial instance: no spins -> energy is the constant offset only.
    if nqubits <= 0
        return Dict(
            "backend" => "generic_tn_tropical",
            "contraction_width" => 0.0,
            "time_complexity" => 0.0,
            "exact_ground_energy" => constant,
            "within_memory_budget" => true,
            "chi_required" => 1.0,
            "nqubits" => 0,
        )
    end

    g, J, h = build_spinglass(nqubits, interactions)
    problem = SpinGlass(g, J, h)

    Random.seed!(seed)
    net = GenericTensorNetwork(problem; optimizer = TreeSA(ntrials = 5, niters = 50))
    cc = contraction_complexity(net)
    sc = Float64(cc.sc)
    tc = Float64(cc.tc)

    within_budget = sc <= memory_sc_budget
    exact_ground = nothing
    if within_budget
        exact_ground = Float64(solve(net, SizeMin())[].n) + constant
    end

    checks = AgentBible.CheckResult[]
    record_validation!(checks, "finite", () -> AgentBible.check_finite(sc; name = "contraction_width"))
    if exact_ground !== nothing
        record_validation!(
            checks, "finite", () -> AgentBible.check_finite(exact_ground; name = "exact_ground_energy")
        )
    end
    append_provenance(checks)

    return Dict(
        "backend" => "generic_tn_tropical",
        "contraction_width" => sc,
        "time_complexity" => tc,
        "exact_ground_energy" => exact_ground,
        "within_memory_budget" => within_budget,
        "chi_required" => 2.0^ceil(sc),
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
