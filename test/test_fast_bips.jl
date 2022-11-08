using BIPs, Statistics, StaticArrays, Random, Test

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

   convert_1pbasis(bR::ChebBasis) = chebyshev_basis(bR.maxn+1)
   convert_1pbasis(bR::TrigBasis) = CTrigBasis(bR.maxL)
   convert_1pbasis(bR::TrigBasisNA) = CTrigBasis(bR.maxL)

   function convert_A_spec(Abasis)
      bR = convert_1pbasis(Abasis.bR)
      bT = convert_1pbasis(Abasis.bT)
      bV = convert_1pbasis(Abasis.bV)
      iR = inv_map(natural_indices(bR))
      iT = inv_map(natural_indices(bT))
      iV = inv_map(natural_indices(bV))
      spec = [ (iR[b.k], iT[b.l], iV[b.n]) for b in Abasis.spec ]
      return spec, (bR, bT, bV)
   end

   function FastBIPf(f_bip_old)
      spec_A, (bR, bT, bV) = convert_A_spec(f_bip_old.Abasis)
      bA = PooledSparseProduct{3}(spec_A)
      return FastBIPf(bR, bT, bV, bA, nothing)
   end


   function eval_A(bipf::FastBIPf, x)
      r = x[1] 
      cθ = x[2]
      sθ = x[3] 
      y = x[4]
      R = bipf.bR(r)
      T = bipf.bT(atan(sθ, cθ))
      V = bipf.bV(y)
      return ACEcore.evaluate(bipf.bA, (R, T, V))
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
R2 = f_bip.Abasis.bR(r)
display(collect(R2) ./ R1)

θ = rand(); sθ, cθ = sincos(θ)
T1 = f_bip_fast.bT(θ)
T2 = f_bip.Abasis.bT(cθ, sθ)
collect(T2) ≈ T1[[5, 3, 1, 2, 4]]

y = rand()
V1 = f_bip_fast.bV(y)
V2 = f_bip.Abasis.bV(y)
collect(V2) ≈ V1[[5, 3, 1, 2, 4]]

## check the A basis now 

A1 = hcat( [X.eval_A(f_bip_fast, x) for x in X1]... )
A2 = hcat( [f_bip.Abasis( [X1[1]] ) for x in X1]... )

s = A2[:, 1] ./ A1[:, 1]
A1 = A1 .* s

A1 ≈ A2