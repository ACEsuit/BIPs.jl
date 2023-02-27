
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

#-------------------- Main BIP embedding layer 

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

# -----------------



struct BIPbasis{T, TR, TT, TV} <: AbstractExplicitContainerLayer{(:bR, :bT, :bV)}
   bR::TR # r basis - k
   bT::TT # θ basis - l
   bV::TV # y basis - n
   bA::PooledSparseProduct{3}
   bAA::SparseSymmProd{Complex{T}}
   maxlen::Int 
end


# ---------- parameter and state management

Base.length(bip::BIPbasis) = length(bip.bAA)

initialparameters(rng::AbstractRNG, bip::BIPbasis) = 
      initialparameters(bip)

initialparameters(bip::BIPbasis) = (
         bR = initialparameters(bip.bR), 
         bT = initialparameters(bip.bT), 
         bV = initialparameters(bip.bV), 
      )


initialstates(rng::AbstractRNG, bip::BIPbasis) = 
         initialstates(bip)

initialstates(bip::BIPbasis{T}) where {T} = (
         A = Vector{Complex{T}}(undef, length(bip.bA)),
         AA = Vector{T}(undef, length(bip.bAA)),
         AAc = Vector{Complex{T}}(undef, length(bip.bAA.dag)),
         # 
         bR = initialstates(bip.bR), 
         bT = initialstates(bip.bT), 
         bV = initialstates(bip.bV), 
      )


# ---------- Conversion from old BIPs 

function inv_collection(a)
   ia = Dict{eltype(a), Int}()
   for (i, ai) in enumerate(a)
      ia[ai] = i 
   end
   return ia 
end

function idx_map(basis)
   a = natural_indices(basis)
   ia = Dict{eltype(a), Int}() 
   for ai in a
      ia[ai] = Polynomials4ML.index(basis, ai)
   end
   return ia 
end

function convert_r_basis(bR::ChebBasis, maxlen)
   Bnew = chebyshev_basis(bR.maxn+1)
   Bnew.A[1:2] .= 1.0 
   Bnew.A[3:end] .= 2 
   Bnew.B[:] .= 0.0 
   Bnew.C[:] .= -1.0 
   r_trans = x -> (log(x[1]) + 4.7) / 6
   l = SimpleRtMEmbedding(Float64, r_trans, Bnew, maxlen) 
   return l, idx_map(Bnew)
end

function convert_θ_basis(bT::Union{TrigBasis, TrigBasisNA}, maxlen) 
   Bnew = CTrigBasis(bT.maxL)
   l = ConstEmbedding(Float64, ComplexF64, 
            x -> atan(x[3], x[2]),
            Bnew, 
            maxlen
         )
   return l, idx_map(Bnew)
end

function convert_y_basis(bT::Union{TrigBasis, TrigBasisNA}, maxlen) 
   Bnew = CTrigBasis(bT.maxL)
   l = ConstEmbedding(Float64, ComplexF64, 
            x -> x[4], 
            Bnew, 
            maxlen
         )
   return l, idx_map(Bnew)
end


function convert_A_spec(Abasis, maxlen)
   bR, iR = convert_r_basis(Abasis.bR, maxlen)
   bT, iT = convert_θ_basis(Abasis.bT, maxlen)
   bV, iV = convert_y_basis(Abasis.bV, maxlen)
   spec = [ (iR[b.k], iT[b.l], iV[b.n]) for b in Abasis.spec ]
   return spec, (bR, bT, bV)
end

function convert_AA_spec(f_bip)
   AA_spec = f_bip.spec
   AA_ords = f_bip.ords
   spec = [ [AA_spec[i, k] for i = 1:AA_ords[k]] for k in 1:size(AA_spec, 2) ]
   spec = sort.(spec)
   return ACEcore.SparseSymmProd(spec; T = ComplexF64)
end

function BIPbasis(f_bip_old; maxlen = 200)
   spec_A, (bR, bT, bV) = convert_A_spec(f_bip_old.Abasis, maxlen)
   bA = PooledSparseProduct{3}(spec_A)
   bAA = convert_AA_spec(f_bip_old)
   return BIPbasis(bR, bT, bV, bA, bAA, maxlen)
end




# ---------- evaluation code 

function (bipf::BIPbasis)(X::AbstractVector{<: SVector}) 
   ps = initialparameters(bipf)
   st = initialstates(bipf)
   return bipf(X, ps, st)[1]
end

# function barrier 
function _eval!(bipf::BIPbasis, X::AbstractVector{<: SVector}, ps, st::NamedTuple)
   R, _ = bipf.bR(X, ps.bR, st.bR)
   T, _ = bipf.bT(X, ps.bT, st.bT)
   Y, _ = bipf.bV(X, ps.bV, st.bV)

   nX = length(X)
   return _eval_inner!(bipf, R, T, Y, st, nX)
end

function _eval_inner!(bipf, R, T, Y, st, nX)
   if bipf.maxlen < nX 
      error("BIPbasis: $nX = nX > maxlen = $(bip.maxlen)")
   end

   A = st.A
   AA = st.AA
   AAc = st.AAc

   # this is the bottleneck!!! 
   ACEcore.evalpool!(A, bipf.bA, (R, T, Y), nX)

   # this circumvents a performance bug in ACEcore 
   ACEcore.evaluate!(AAc, bipf.bAA.dag, A)
   @inbounds @simd ivdep for i = 1:length(bipf.bAA)
      AA[i] = real(AAc[bipf.bAA.proj[i]])
   end

   return AA
end


function (bipf::BIPbasis)(X::AbstractVector{<: SVector}, ps::NamedTuple, st::NamedTuple)
   AA = _eval!(bipf, X, ps, st)
   return AA, st 
end


import ChainRulesCore: rrule, NoTangent, ZeroTangent, NoTangent

function rrule(::typeof(_eval_inner!), bipf::BIPbasis, R_, T_, Y_, st, nX)
   R = copy(collect(R_)) 
   T = copy(collect(T_))
   Y = copy(collect(Y_))
   A = st.A |> copy 
   AA = st.AA |> copy 
   AAc = st.AAc |> copy 

   # layer 1: R, T, Y -> A
   ACEcore.evalpool!(A, bipf.bA, (R, T, Y), nX)

   # layer 2: A -> AAc 
   ACEcore.evaluate!(AAc, bipf.bAA.dag, A)
   map!(real, AAc, AAc)

   # layer 3: AAc -> AA
   @inbounds @simd ivdep for i = 1:length(bipf.bAA)
      AA[i] = real(AAc[bipf.bAA.proj[i]])
   end

   function pb(ΔAA)
      # 3: pullback from AA to AAc 
      ΔAAc = zeros(ComplexF64, length(st.AAc))
      ΔAAc[bipf.bAA.proj] = ΔAA

      # 2: pullback from AAc to A
      ΔA = zeros(ComplexF64, size(st.A))
      ACEcore.pullback_arg!(ΔA, ΔAAc, bipf.bAA.dag, AAc) 

      # 1: pullback from A to (R, T, Y)
      ΔR, ΔT, ΔY = ACEcore._pullback_evalpool(ΔA, bipf.bA, (R, T, Y))

      return NoTangent(), NoTangent(), ΔR, ΔT, ΔY, NoTangent(), NoTangent()
   end

   return AA, pb
end


# -------------- Interface 


function simple_chebyshev(maxn)
   cheb = chebyshev_basis(maxn+1)
   cheb.A[1:2] .= 1.0 
   cheb.A[3:end] .= 2 
   cheb.B[:] .= 0.0 
   cheb.C[:] .= -1.0 
   return cheb
end



function simple_bips(; order = 3, maxlevel = 6, n_pt = 5, n_th = 3, n_y = 3, 
                       maxlen = 200)

   # radial embedding : this also incorporate the * tM operation
   cheb = simple_chebyshev(n_pt)
   inds_pt = natural_indices(cheb)
   inv_pt = idx_map(cheb)
   pt_trans = x -> (log(x[1]) + 4.7) / 6
   bR = SimpleRtMEmbedding(Float64, pt_trans, cheb, maxlen) 

   # angular embedding 
   trig_θ = CTrigBasis(n_th)
   inds_θ = natural_indices(trig_θ)
   inv_θ = idx_map(trig_θ)
   bT = ConstEmbedding(Float64, ComplexF64, 
                          x -> atan(x[3], x[2]), trig_θ, maxlen)

   # y embedding 
   trig_y = CTrigBasis(n_y)
   inds_y = natural_indices(trig_y)
   inv_y = idx_map(trig_y)
   bY = ConstEmbedding(Float64, ComplexF64, 
                          x -> x[4], trig_y, maxlen)

   # generate a specification 
   Bsel = BIPs.BiPolynomials.Modules.BasisSelector(; 
            order = order, levels = maxlevel)
   spec_A, levels = BIPs.BiPolynomials.generate_spec_A(inds_pt, inds_θ, inds_y, Bsel)
   spec_AA = BIPs.BiPolynomials.generate_spec_AA(spec_A, levels, Bsel)

   # generate the one-particle basis 
   spec_A_2 = [ (inv_pt[b.k], inv_θ[b.l], inv_y[b.n]) for b in spec_A ]
   bA = PooledSparseProduct{3}(spec_A_2)

   # ... and the AA basis 
   spec_AA = sort.(spec_AA)   
   bAA = SparseSymmProd(spec_AA; T = ComplexF64)

   # put it all together 
   return ConstL(BIPbasis(bR, bT, bY, bA, bAA, maxlen))
end

function simple_bips2(; order = 3, maxlevel = 6, n_pt = 5, n_th = 3, n_y = 3, 
                        pt_trans = x -> (log(x[1]) + 4.7) / 6,
                        maxlen = 200)
   # radial embedding : this also incorporate the * tM operation
   bR = simple_transverse_embedding(; pt_trans = pt_trans, n_pt = n_pt, 
                                 maxlen = maxlen)   
   # angular embedding 
   bT = angular_embedding(; n_th = n_th, maxlen = maxlen)
   # y embedding 
   bY = y_embedding(; n_y = n_y, maxlen = maxlen)

   return bips2(bR, bT, bY; order = order, maxlevel = maxlevel)
end


function bips2(tB, θB, yB; order = 3, maxlevel = 6)

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




function bips(tB, θB, yB; order = 3, maxlevel = 6, maxlen = 200)

   inds_pt = tB.meta["inds"]
   inv_pt = tB.meta["inv"]
   inds_θ = θB.meta["inds"]
   inv_θ = θB.meta["inv"]
   inds_y = yB.meta["inds"]
   inv_y = yB.meta["inv"]

   # generate a specification 
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
   return BIPbasis(tB, θB, yB, bA, bAA, maxlen)   
end


end

