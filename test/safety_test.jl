
using BIPs, Statistics, StaticArrays, Random

f_bip, specs = build_ip(order=3,
    levels=5,
    n_pt=4,
    n_th=2,
    n_y=2)

hyp_jets = sample_hyp_jets    


##    

@info("IR Safety test")

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


n_limit = 1_000
ir_added_jets = [add_ir(jet) for jet in hyp_jets]
difference_in_data = all(~any(hyp_jets .== ir_added_jets))
println(@test difference_in_data)

emb_jets = bip_data(hyp_jets)
ir_added_emb_jets = bip_data(ir_added_jets)
invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- ir_added_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])
println(@test invariance_in_embedding)


##

@info("Collinear Safety test")

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

n_limit = 1_000
collinear_added_jets = [add_collinear(jet) for jet in hyp_jets]
difference_in_data = all(~any(hyp_jets .== collinear_added_jets))
println(@test difference_in_data)

emb_jets = bip_data(hyp_jets)
collinear_added_emb_jets = bip_data(collinear_added_jets)
invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- collinear_added_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])
println(@test invariance_in_embedding)