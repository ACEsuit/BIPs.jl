using BIPs, Statistics, StaticArrays, Random, Test

include("testing_tools.jl")
hyp_jets = sample_hyp_jets


f_bip, specs = build_ip(order=3,
    levels=5,
    n_pt=4,
    n_th=2,
    n_y=2)

##

@info("Boost invariance test")

function boostJet(jet; β=rand())
    boosted_jet = []
    for particle in jet
        push!(boosted_jet, SVector{5}([particle[1],
            particle[2],
            particle[3],
            particle[4] + atanh(β),
            particle[5]]))
    end
    boosted_jet
end


n_limit = 1_000
all_boosted = [boostJet(jet) for jet in hyp_jets]
difference_in_data = all(~any(hyp_jets .== all_boosted))
println(@test difference_in_data)

emb_jets = bip_data(hyp_jets)
boosted_emb_jets = bip_data(all_boosted)
invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- boosted_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])
println(@test invariance_in_embedding)

##

@info("Permutation invariance test")

function permute(jet)
    modified_jet = shuffle(jet)
    modified_jet
end

n_limit = 1_000
permuted_jets = [permute(jet) for jet in hyp_jets]
difference_in_data = all(~any(hyp_jets .== permuted_jets))
println(@test difference_in_data)

emb_jets = bip_data(hyp_jets)
permuted_emb_jets = bip_data(permuted_jets)
invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- permuted_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])
println(@test invariance_in_embedding)
