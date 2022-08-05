module DataTransformer
using Statistics, LinearAlgebra, StaticArrays

"""Aligns a particle to a jet.
Args:
    particle (np.array):  The particle to align.
    unit_p (np.array):  The jet to align to.
Returns:
    tuple: Energy, Parallel  momentum, and Perpendicular momentum.
"""
function transform2hyp!(tjet, jet::Vector{<:SVector}; ϵ=1e-10)
    pbar4 = mean(jet)
    pbar3 = SVector(pbar4[2], pbar4[3], pbar4[4])
    pbar_norm = pbar3 / norm(pbar3)
    v = pbar_norm - SVector(0.0, 0.0, 1.0)
    norm_v = norm(v)
    if !isapprox(norm_v, 0.0)
        v = v / norm_v
    end
    function transform(Ep, v)
        E = Ep[1]
        p = SVector(Ep[2], Ep[3], Ep[4])
        xyz = p - 2 * dot(p, v) * v
        z = xyz[3]
        tM = log((E^2 - z^2 + 1)^0.5)
        y = 0.5 * log((E + z + ϵ) / (E - z + ϵ))
        r = sqrt(xyz[1]^2 + xyz[2]^2) + ϵ
        cosθ, sinθ = xyz[1] / r, xyz[2] / r
        return SVector(r, cosθ, sinθ, y, tM)
    end

    for i = 1:length(jet)
        @inbounds tjet[i] = transform(jet[i], v)
    end

    # Normalization
    Vec_tjet = reinterpret(Float64, tjet)
    sum_ = sum(x[1] for x in tjet)
    for i in 1:5:length(Vec_tjet)
        @inbounds Vec_tjet[i] /= (sum_ + ϵ)
    end
    return tjet
end

transform2hyp(jet::Vector{<:SVector}; ϵ=1e-4) =
    transform2hyp!(Vector{SVector{5,Float64}}(undef, length(jet)),
        jet; ϵ=ϵ)

function data2hyp(dataset_jets)
    storage = Vector{SVector}[]
    for i = eachindex(dataset_jets)
        push!(storage, transform2hyp(dataset_jets[i]))
    end
    storage
end

export transform2hyp, data2hyp


end