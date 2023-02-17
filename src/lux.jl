
module LuxBIPs

using Polynomials4ML, ACEcore, StaticArrays
using ACEcore: PooledSparseProduct, SparseSymmProd
using Polynomials4ML: natural_indices
using BIPs.BiPolynomials.Modules: TrigBasis, TrigBasisNA, ChebBasis
using LinearAlgebra: Diagonal, mul!
using ObjectPools
using ObjectPools: acquire!, release!


struct BIPbasis{TR, TT, TV} 
   bR::TR # r basis - k
   bT::TT # θ basis - l
   bV::TV # y basis - n
   bA::PooledSparseProduct{3}
   bAA::SparseSymmProd{ComplexF64}
   # ---------------- Temporaries
   AAc::ArrayCache{ComplexF64, 1}
   AA::ArrayCache{Float64, 1}
end

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
   return BIPbasis(bR, bT, bV, bA, bAA, 
                   ArrayCache(ComplexF64, 1), 
                   ArrayCache(Float64, 1) )
end


# ---------- memory management code 


# ---------- evaluation code 


function _addinto!(A, bipf::BIPbasis, x)
   r = (log(x[1]) + 4.7) / 6 # x[1] 
   cθ = x[2]
   sθ = x[3] 
   y = x[4]
   R = bipf.bR(r) * x[end] 
   T = bipf.bT(atan(sθ, cθ))
   V = bipf.bV(y)
   A[:] .= A[:] .+ ACEcore.evaluate(bipf.bA, (R, T, V))
end


function (bipf::BIPbasis)(X::AbstractVector{<: SVector})
   # A = zeros(ComplexF64, length(bipf.bA))
   # for x in X
   #    _addinto!(A, bipf, x)
   # end

   r = [ (log(x[1]) + 4.7) / 6 for x in X ] 
   θ = [ atan(x[3], x[2]) for x in X ]
   y = [ x[4] for x in X ]
   tM = [ x[5] for x in X ]

   R = Polynomials4ML.evaluate(bipf.bR, r) 
   mul!(R, Diagonal(tM), R)

   T = bipf.bT(θ)
   V = bipf.bV(y)

   A = ACEcore.evalpool(bipf.bA, (R, T, V))

   Polynomials4ML.release!(R)
   Polynomials4ML.release!(T)
   Polynomials4ML.release!(V)

   # this circumvents a performance bug in ACEcore 
   AAc = acquire!(bipf.AAc, length(bipf.bAA.dag))
   ACEcore.evaluate!(parent(AAc), bipf.bAA.dag, A)
   AA = acquire!(bipf.AA, length(bipf.bAA)) 
   AA_ = parent(AA)
   AA_[1] = 1  # [hack] 
   @inbounds @simd for i = 2:length(bipf.bAA)
      AA_[i] = real(AAc[bipf.bAA.proj[i]])
   end
   # AA = [ real(AAc[ bipf.bAA.proj[i] ]) for i = 1:length(bipf.bAA) ]
   # AA[1] = 1    # [hack] 

   ACEcore.release!(A)
   ACEcore.release!(AAc)   

   return AA
end



end