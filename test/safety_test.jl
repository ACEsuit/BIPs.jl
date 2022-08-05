module PhysicsSafetyTester
using BIPs
using Statistics, StaticArrays, Random

f_bip, specs = build_ip(order=3,
    levels=5,
    n_pt=4,
    n_th=2,
    n_y=2)

function bip_data(dataset_jets)
    storage = zeros(length(dataset_jets), length(specs))
    for i = 1:length(dataset_jets)
        storage[i, :] = f_bip(dataset_jets[i])
    end
    storage
end

function ir_safety_test(hyp_jets)
    n_limit = 1_000
    function add_ir(jet; δ=1e-15)
        modified_jet = copy(jet)
        θ = rand() * 2 * pi
        ϕ = rand() * pi
        E = δ
        pp = E * rand()
        r = pp * sin(ϕ)
        pL = pp * cos(ϕ)
        tM = log((E^2 - pL^2 + 1)^0.5)
        ϵ = 0.01
        y = log((E + pL + ϵ) / (E - pL + ϵ))
        particle = @SVector [r, cos(θ), sin(θ), y, tM]
        push!(modified_jet, particle)
        modified_jet
    end
    ir_added_jets = [add_ir(jet) for jet in hyp_jets]
    difference_in_data = all(~any(hyp_jets .== ir_added_jets))
    @show difference_in_data

    emb_jets = bip_data(hyp_jets)
    ir_added_emb_jets = bip_data(ir_added_jets)

    invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- ir_added_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])
    return difference_in_data && invariance_in_embedding
end


function collinear_safety_test(hyp_jets)
    n_limit = 1_000
    function add_collinear(jet; δ=1e-15)
        modified_jet = copy(jet)
        θ = rand() * 2 * pi
        r = rand() * δ
        E = rand()
        pL = E * (1 - δ)
        tM = log((E^2 - pL^2 + 1)^0.5)
        ϵ = 0.01
        y = log((E + pL + ϵ) / (E - pL + ϵ))
        particle = @SVector [r, cos(θ), sin(θ), y, tM]
        push!(modified_jet, particle)
    end

    collinear_added_jets = [add_collinear(jet) for jet in hyp_jets]
    difference_in_data = all(~any(hyp_jets .== collinear_added_jets))

    emb_jets = bip_data(hyp_jets)
    collinear_added_emb_jets = bip_data(collinear_added_jets)

    invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- collinear_added_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])

    return difference_in_data && invariance_in_embedding

end

export collinear_safety_test, ir_safety_test
end