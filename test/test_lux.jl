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

using  BenchmarkTools, LuxCore
X = identity.(hyp_jets[1])

@btime $f_bip($X)
@btime $f_bip_lux($X)

rng = MersenneTwister(1234)
ps, st = LuxCore.setup(rng, f_bip_lux)
# = LuxCore.initialparameters(f_bip_lux)
# st = LuxCore.initialstates(f_bip_lux)
@btime $f_bip_lux($X, $ps, $st)


## 

using Lux 

model = Chain(; bip = f_bip_lux, 
                l1 = Dense(length(f_bip_lux), 10), 
                l2 = Dense(10, 10, tanh), 
                l3 = Dense(10, 1) )

rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, model)              

model(X, ps, st)[1] 

@btime $model($X, $ps, $st)