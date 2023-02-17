using BIPs, Statistics, StaticArrays, Random, Test, ACEcore, 
      Polynomials4ML, LinearAlgebra
using Polynomials4ML.Testing: print_tf      

include("testing_tools.jl")
hyp_jets = sample_hyp_jets

##

f_bip, specs = build_ip(order=3,
      levels=6,
      n_pt=4,
      n_th=2,
      n_y=2)

f_bip_lux = BIPs.LuxBIPs.BIPbasis(f_bip)

##

for X in hyp_jets[1:100]
   AA1 = f_bip(X)
   AA2 = f_bip_lux(X)
   print_tf(@test norm((AA1 - AA2) ./ abs.(AA2), Inf) < 0.1)
   print_tf(@test norm(AA1 - AA2, Inf) / norm(AA2, Inf) < 1e-4)
end

##

using  BenchmarkTools, ObjectPools
X = identity.(hyp_jets[1])

@btime $f_bip($X)
@btime (B = $f_bip_lux($X); release!(B); nothing)


##

@profview let X = X, f_bip_lux = f_bip_lux
   for _ = 1:1_000_000 
      f_bip_lux(X)
   end
end

##

@code_warntype f_bip_lux(X)

