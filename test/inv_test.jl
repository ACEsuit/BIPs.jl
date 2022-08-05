module InvarianceTester
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

function boost_invariance_test(hyp_jets)
    n_limit = 1_000
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
    all_boosted = [boostJet(jet) for jet in hyp_jets]
    difference_in_data = all(~any(hyp_jets .== all_boosted))

    emb_jets = bip_data(hyp_jets)
    boosted_emb_jets = bip_data(all_boosted)

    invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- boosted_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])

    return difference_in_data && invariance_in_embedding

end


function permutation_invariance_test(hyp_jets)
    n_limit = 1_000
    function permute(jet)
        modified_jet = shuffle(jet)
        modified_jet
    end
    permuted_jets = [permute(jet) for jet in hyp_jets]
    difference_in_data = all(~any(hyp_jets .== permuted_jets))

    emb_jets = bip_data(hyp_jets)
    permuted_emb_jets = bip_data(permuted_jets)

    invariance_in_embedding = all([abs(mean(emb_jets[i, :] .- permuted_emb_jets[i, :])) < 1e-5 for i in 1:n_limit])
    return difference_in_data && invariance_in_embedding
end

export permutation_invariance_test, boost_invariance_test
end