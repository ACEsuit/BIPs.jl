
module LuxBIPs

using Polynomials4ML, ACEcore, StaticArrays
using ACEcore: PooledSparseProduct, SparseSymmProd
using Polynomials4ML: natural_indices
using BIPs.BiPolynomials.Modules: TrigBasis, TrigBasisNA, ChebBasis
using LinearAlgebra: Diagonal, mul!
using ObjectPools
using ObjectPools: acquire!, release!

using LuxCore 
import LuxCore: initialparameters, initialstates, 
                  AbstractExplicitLayer

# for now assume that bR, bT, bV, bA, bAA are not layers, i.e. they
# can't have parameters. 

struct BIPbasis{T, TR, TT, TV} <: AbstractExplicitLayer
   bR::TR # r basis - k
   bT::TT # θ basis - l
   bV::TV # y basis - n
   bA::PooledSparseProduct{3}
   bAA::SparseSymmProd{Complex{T}}
   maxlen::Int 
end

# ---------- parameter and state management

initialparameters(::BIPbasis) = NamedTuple()

initialstates(bip::BIPbasis{T}) where {T} = (
         r = Vector{T}(undef, bip.maxlen), 
         θ = Vector{T}(undef, bip.maxlen),
         y = Vector{T}(undef, bip.maxlen),
         tM = Vector{T}(undef, bip.maxlen),
         R = Matrix{T}(undef, bip.maxlen, length(bip.bR)),
         T = Matrix{Complex{T}}(undef, bip.maxlen, length(bip.bT)),
         V = Matrix{Complex{T}}(undef, bip.maxlen, length(bip.bV)),
         A = Vector{Complex{T}}(undef, length(bip.bA)),
         AA = Vector{T}(undef, length(bip.bAA)),
         AAc = Vector{Complex{T}}(undef, length(bip.bAA.dag)),
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

function convert_1pbasis(bR::ChebBasis) 
   Bnew = chebyshev_basis(bR.maxn+1)
   Bnew.A[1:2] .= 1.0 
   Bnew.A[3:end] .= 2 
   Bnew.B[:] .= 0.0 
   Bnew.C[:] .= -1.0 
   return Bnew 
end

convert_1pbasis(bR::TrigBasis) = CTrigBasis(bR.maxL)

convert_1pbasis(bR::TrigBasisNA) = CTrigBasis(bR.maxL)

function convert_A_spec(Abasis)
   bR = convert_1pbasis(Abasis.bR)
   bT = convert_1pbasis(Abasis.bT)
   bV = convert_1pbasis(Abasis.bV)
   iR = idx_map(bR)
   iT = idx_map(bT)
   iV = idx_map(bV)
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

function BIPbasis(f_bip_old)
   spec_A, (bR, bT, bV) = convert_A_spec(f_bip_old.Abasis)
   bA = PooledSparseProduct{3}(spec_A)
   bAA = convert_AA_spec(f_bip_old)
   return BIPbasis(bR, bT, bV, bA, bAA, 200)
end




# ---------- evaluation code 

function (bipf::BIPbasis)(X::AbstractVector{<: SVector}) 
   ps = initialparameters(bipf)
   st = initialstates(bipf)
   return bipf(X, ps, st)[1]
end

function (bipf::BIPbasis)(X::AbstractVector{<: SVector}, ps::NamedTuple, st::NamedTuple)
   r = st.r 
   θ = st.θ
   y = st.y
   tM = st.tM
   R = st.R
   T = st.T
   V = st.V
   A = st.A
   AA = st.AA
   AAc = st.AAc

   nX = length(X)
   if bipf.maxlen < nX 
      error("BIPbasis: $nX = nX > maxlen = $(bip.maxlen)")
   end

   @inbounds @simd for i = 1:nX
      x = X[i] 
      r[i] = (log(x[1]) + 4.7) / 6
      θ[i] = atan(x[3], x[2])
      y[i] = x[4]
      tM[i] = x[5]
   end

   Polynomials4ML.evaluate!(R, bipf.bR, (@view r[1:nX]))
   Polynomials4ML.evaluate!(T, bipf.bT, (@view θ[1:nX]))
   Polynomials4ML.evaluate!(V, bipf.bV, (@view y[1:nX]))

   # rescale with transverse momentum 
   @inbounds for j = 1:size(R, 2)
      @simd ivdep for i = 1:nX
         R[i, j] *= tM[i]
      end
   end

   # this is the bottleneck!!! 
   ACEcore.evalpool!(A, bipf.bA, (R, T, V), nX)

   # this circumvents a performance bug in ACEcore 
   ACEcore.evaluate!(AAc, bipf.bAA.dag, A)
   AA[1] = 1  # [hack]  ACEcore doesn't allow the constant basis function :(
   @inbounds @simd ivdep for i = 2:length(bipf.bAA)
      AA[i] = real(AAc[bipf.bAA.proj[i]])
   end

   return AA, st 
end



end