
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

include("embedding_layers.jl") 


#-------------------- Main BIP embedding layer 

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
   l = GenericEmbedding(Float64, ComplexF64, 
            x -> atan(x[3], x[2]),
            Bnew, 
            maxlen
         )
   return l, idx_map(Bnew)
end

function convert_y_basis(bT::Union{TrigBasis, TrigBasisNA}, maxlen) 
   Bnew = CTrigBasis(bT.maxL)
   l = GenericEmbedding(Float64, ComplexF64, 
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
   spec = [ [AA_spec[i, k] for i = 1:AA_ords[k]] for k in 2:size(AA_spec, 2) ]
   spec = sort.(spec)
   spec = [ [spec[1],]; spec ]  # duplicate the first feature; this is a [hack] explained below 
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
   A = st.A
   AA = st.AA
   AAc = st.AAc

   nX = length(X)
   if bipf.maxlen < nX 
      error("BIPbasis: $nX = nX > maxlen = $(bip.maxlen)")
   end

   R = bipf.bR(X, ps.bR, st.bR)
   T = bipf.bT(X, ps.bT, st.bT)
   V = bipf.bV(X, ps.bV, st.bV)

   # this is the bottleneck!!! 
   ACEcore.evalpool!(A, bipf.bA, (R, T, V), nX)

   # this circumvents a performance bug in ACEcore 
   ACEcore.evaluate!(AAc, bipf.bAA.dag, A)
   AA[1] = 1  # [hack]  ACEcore doesn't allow the constant basis function :(
   @inbounds @simd ivdep for i = 2:length(bipf.bAA)
      AA[i] = real(AAc[bipf.bAA.proj[i]])
   end

   return AA 
end

function (bipf::BIPbasis)(X::AbstractVector{<: SVector}, ps::NamedTuple, st::NamedTuple)
   # AA = Zygote.ignore() do
   #    _eval!(bipf, X, ps, st)
   # end
   AA = _eval!(bipf, X, ps, st)
   return AA, st 
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
   bT = GenericEmbedding(Float64, ComplexF64, 
                          x -> atan(x[3], x[2]), trig_θ, maxlen)

   # y embedding 
   trig_y = CTrigBasis(n_y)
   inds_y = natural_indices(trig_y)
   inv_y = idx_map(trig_y)
   bY = GenericEmbedding(Float64, ComplexF64, 
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
   @assert length(spec_AA[1]) == 0 
   # replace this one with a duplicate, this is a hack needed because 
   # the current AA basis evaluator doesn't admit, this is fixed in the 
   # evaluator code. 
   spec_AA[1] = copy(spec_AA[2])
   spec_AA = sort.(spec_AA)   
   bAA = SparseSymmProd(spec_AA; T = ComplexF64)

   # put it all together 
   return BIPbasis(bR, bT, bY, bA, bAA, maxlen)
end


end

