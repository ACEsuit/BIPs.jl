
module LuxBIPs

using Polynomials4ML, ACEcore, StaticArrays
using ACEcore: PooledSparseProduct, SparseSymmProd
using Polynomials4ML: natural_indices
using BIPs.BiPolynomials.Modules: TrigBasis, TrigBasisNA, ChebBasis
using LinearAlgebra: Diagonal, mul!
using Random: AbstractRNG
import Zygote, BIPs

using LuxCore 
import LuxCore: initialparameters, initialstates, 
                  AbstractExplicitLayer, 
                  AbstractExplicitContainerLayer

using Lux: BranchLayer, Chain, WrappedFunction               

using ChainRulesCore: ignore_derivatives

include("embedding_layers.jl") 

#-------------------- an auxiliary layer...

"""
`MetaLayer` just wraps a Lux Layer into another layer and adds MetaData 
information. 
"""
struct MetaLayer{TL} <: AbstractExplicitLayer
   l::TL
   meta::Dict{String, Any}
end

MetaLayer(l) = MetaLayer(l, Dict{String, Any}())

Base.length(l::MetaLayer) = length(l.l)

initialparameters(rng::AbstractRNG, l::MetaLayer) = 
      (l = initialparameters(rng, l.l), )

initialstates(rng::AbstractRNG, l::MetaLayer) = 
      (l = initialstates(rng, l.l), )
      
(l::MetaLayer)(x, ps, st) = l.l(x, ps.l, st.l)



# -------------- Interface 


function simple_bips(; order = 3, maxlevel = 6, n_pt = 5, n_th = 3, n_y = 3, 
                        pt_trans = x -> (log(x[1]) + 4.7) / 6,
                        maxlen = 200)
   # radial embedding : this also incorporate the * tM operation
   bR = simple_transverse_embedding(; pt_trans = pt_trans, n_pt = n_pt, 
                                 maxlen = maxlen)   
   # angular embedding 
   bT = angular_embedding(; n_th = n_th, maxlen = maxlen)
   # y embedding 
   bY = y_embedding(; n_y = n_y, maxlen = maxlen)

   return bips(bR, bT, bY; order = order, maxlevel = maxlevel)
end


function bips(tB, θB, yB; order = 3, maxlevel = 6)

   # generate a specification 
   inds_pt = tB.meta["inds"]
   inv_pt = tB.meta["inv"]
   inds_θ = θB.meta["inds"]
   inv_θ = θB.meta["inv"]
   inds_y = yB.meta["inds"]
   inv_y = yB.meta["inv"]

   Bsel = BIPs.BiPolynomials.Modules.BasisSelector(; 
                     order = order, levels = maxlevel)
   spec_A, levels = BIPs.BiPolynomials.generate_spec_A(inds_pt, inds_θ, inds_y, Bsel)
   spec_AA = BIPs.BiPolynomials.generate_spec_AA(spec_A, levels, Bsel)
   spec_AA = sort.(spec_AA)

   # generate the one-particle basis 
   spec_A_2 = [ (inv_pt[b.k], inv_θ[b.l], inv_y[b.n]) for b in spec_A ]
   bA = PooledSparseProduct{3}(spec_A_2) 

   # ... and the AA basis 
   bAA = SparseSymmProd(spec_AA; T = ComplexF64)

   # put it all together 
   embed = BranchLayer((tB = tB, θB = θB, yB = yB,))
   pool = ACEcore.lux(bA) 
   corr = ACEcore.lux(bAA)
   f_bip = Chain(; embed = embed, pool = pool, corr = corr, 
                   real = WrappedFunction(real) )
   return f_bip
end



# ----------------- Extract human-readable specification 

function get_embedding_spec(l)
   invi = l.meta["inv"]
   @show invi 
   spec = Vector{Any}(undef, length(invi))
   for k in keys(invi) 
      spec[invi[k]] = k
   end
   return identity.(spec)
end      


function get_1p_spec(bip)
   tB = bip.layers.embed.layers.tB   # n
   t_spec = get_embedding_spec(tB)
   yB = bip.layers.embed.layers.yB   # k 
   y_spec = get_embedding_spec(yB)
   θB = bip.layers.embed.layers.θB   # l 
   θ_spec = get_embedding_spec(θB)

   spec_A_ = bip.layers.pool.basis.spec
   spec_A = [ (n = t_spec[b[1]], l = θ_spec[b[2]], k = y_spec[b[3]]) for b in spec_A_ ]

   return spec_A
end

function get_bip_spec(bip)
   spec_A = get_1p_spec(bip)
   spec_AA_ = ACEcore.reconstruct_spec(bip.layers.corr.basis)
   spec_AA = [ (length(bb) == 0 ? Vector{eltype(spec_A)}(undef, 0) 
                               :  [ spec_A[bb[i]] for i = 1:length(bb) ] )
              for bb in spec_AA_]
   return spec_AA            
end


end

