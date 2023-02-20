
module LuxBIPs

using Polynomials4ML, ACEcore, StaticArrays
using ACEcore: PooledSparseProduct, SparseSymmProd
using Polynomials4ML: natural_indices
using BIPs.BiPolynomials.Modules: TrigBasis, TrigBasisNA, ChebBasis
using LinearAlgebra: Diagonal, mul!
using Random: AbstractRNG
import Zygote

using LuxCore 
import LuxCore: initialparameters, initialstates, 
                  AbstractExplicitLayer, 
                  AbstractExplicitContainerLayer

# ----------- a simple embedding interface 
#             so that We can give make learnable embeddings 

# careful, this assumes that transform is not trainable 
# This Lux wrapper should go into ACEbase or Polynomials4ML 
# but for now this is a convenient prototype implementation
struct GenericEmbedding{TIN, TOUT, TT, TB} <: AbstractExplicitLayer 
   transform::TT
   B::TB
   maxlen::Int 
end

GenericEmbedding(TIN, TOUT, transform, B, maxlen) = 
      GenericEmbedding{TIN, TOUT, typeof(transform), typeof(B)}(
         transform, B, maxlen
      )

Base.length(l::GenericEmbedding) = length(l.B)

function (l::GenericEmbedding)(X, ps, st)
   nX = length(X) 
   x = st.x  
   P = st.P 
   @assert length(x) >= nX
   @assert size(P, 1) >= nX

   # transform input to correct format 
   @simd ivdep for i = 1:nX 
      @inbounds x[i] = l.transform(X[i])
   end
   # now evaluate the embedding 
   Polynomials4ML.evaluate!(P, l.B, (@view x[1:nX]))
   return P 
end

initialparameters(rng::AbstractRNG, l::GenericEmbedding) = 
      initialparameters(l) 

initialparameters(l::GenericEmbedding) = 
         NamedTuple() 

initialstates(rng::AbstractRNG, l::GenericEmbedding) = 
         initialstates(l) 

initialstates(l::GenericEmbedding{TIN, TOUT}) where {TIN, TOUT} = (
         x = Vector{TIN}(undef, l.maxlen), 
         P = Matrix{TOUT}(undef, l.maxlen, length(l.B))
      )
         

# ---------------- BIP Radial embedding Layer - simple version 

struct SimpleRtMEmbedding{T, TRT, TR} <: AbstractExplicitLayer
   r_trans::TRT
   r_embed::TR
   maxlen::Int 
end

SimpleRtMEmbedding(T, r_trans, r_embed, maxlen) = 
      SimpleRtMEmbedding{T, typeof(r_trans), typeof(r_embed)}(r_trans, r_embed, maxlen)

initialparameters(rng::AbstractRNG, l::SimpleRtMEmbedding) = 
      initialparameters(l) 

initialparameters(l::SimpleRtMEmbedding) = 
      NamedTuple()
         
initialstates(rng::AbstractRNG, l::SimpleRtMEmbedding) = 
         initialstates(l) 

initialstates(l::SimpleRtMEmbedding{T}) where {T} = (
      r = Vector{T}(undef, l.maxlen), 
      tM = Vector{T}(undef, l.maxlen), 
      R = Matrix{T}(undef, l.maxlen, length(l.r_embed))
   )

function (l::SimpleRtMEmbedding)(X, ps, st)
   nX = length(X) 
   r = @view st.r[1:nX]
   tM = st.tM 
   R = st.R 
   @assert size(R, 1) >= nX 
   @assert length(r) >= nX 
   @assert length(tM) >= nX 
   @assert size(R, 2) >= length(l.r_embed)

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         x = X[i] 
         # l.r_trans(x)   # log(x[1] + 4.7) / 6
         r[i] = log(x[1] + 4.7) / 6 
         tM[i] = x[5] 
      end

      # evaluate the r embedding 
      Polynomials4ML.evaluate!(R, l.r_embed, r)

      # apply the tM rescaling 
      @inbounds for j = 1:length(l.r_embed)
         @simd ivdep for i = 1:nX
            R[i, j] *= tM[i]
         end
      end

   end

   return R 
end


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
   rtrans = x -> log(x[1] + 4.7) / 6
   l = SimpleRtMEmbedding(Float64, rtrans, Bnew, maxlen) 
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
   AA = Zygote.ignore() do
      _eval!(bipf, X, ps, st)
   end
   return AA, st 
end

end