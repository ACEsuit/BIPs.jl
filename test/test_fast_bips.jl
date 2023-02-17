using BIPs, Statistics, StaticArrays, Random, Test, ACEcore, 
      Polynomials4ML, LinearAlgebra

include("testing_tools.jl")
hyp_jets = sample_hyp_jets

##

module X

   using Polynomials4ML, ACEcore
   using ACEcore: PooledSparseProduct 
   using Polynomials4ML: natural_indices
   using BIPs.BiPolynomials.Modules: TrigBasis, TrigBasisNA,  ChebBasis

   struct FastBIPf{TR, TT, TV} 
      bR::TR # r basis - k
      bT::TT # θ basis - l
      bV::TV # y basis - n
      bA::PooledSparseProduct{3}
      bAA::Nothing
      # ---------------- Temporaries 
   end

   function inv_map(a)
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

   convert_1pbasis(bR::ChebBasis) = chebyshev_basis(bR.maxn+1)
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

   function FastBIPf(f_bip_old)
      spec_A, (bR, bT, bV) = convert_A_spec(f_bip_old.Abasis)
      bA = PooledSparseProduct{3}(spec_A)
      return FastBIPf(bR, bT, bV, bA, nothing)
   end


   function eval_A(bipf::FastBIPf, x)
      r = (log(x[1]) + 4.7) / 6 # x[1] 
      cθ = x[2]
      sθ = x[3] 
      y = x[4]
      R = bipf.bR(r)
      R *= 1.25331410190992
      R[1] *= 1.4142135823485564
      T = bipf.bT(atan(sθ, cθ))
      V = bipf.bV(y)
      return ACEcore.evaluate(bipf.bA, (R, T, V)) * x[end] 
   end

end

##

f_bip, specs = build_ip(order=3,
      levels=6,
      n_pt=4,
      n_th=2,
      n_y=2)

X1 = hyp_jets[1]    

f_bip.Abasis(X1)

## convert the basis 

f_bip_fast = X.FastBIPf(f_bip)

## check the 1p basis components 

r = rand() 
R1 = f_bip_fast.bR(r)
R1 *= 1.25331410190992; R1[1] *= 1.4142135823485564
R2 = f_bip.Abasis.bR(r)
@show R1 ≈ collect(R2)

θ = rand(); sθ, cθ = sincos(θ)
T1 = f_bip_fast.bT(atan(sθ, cθ))
T2 = f_bip.Abasis.bT(cθ, sθ)
inds = natural_indices(f_bip_fast.bT)
IT = X.idx_map(f_bip_fast.bT)
all(T2[k] ≈ T1[IT[k]] for k in inds)

y = rand()
V1 = f_bip_fast.bV(y)
V2 = f_bip.Abasis.bV(y)
inds = natural_indices(f_bip_fast.bV)
IV = X.idx_map(f_bip_fast.bV)
all(V2[k] ≈ V1[IV[k]] for k in inds)

## check the A basis now 

A1 = hcat( [X.eval_A(f_bip_fast, x) for x in X1]... )
A2 = hcat( [f_bip.Abasis( [x,] ) for x in X1]... )

norm(A1 - A2, Inf)

