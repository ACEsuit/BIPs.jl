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

# benchmark when we pre-allocate
rng = MersenneTwister(1234)
ps, st = LuxCore.setup(rng, f_bip_lux)
@btime $f_bip_lux($X, $ps, $st)


## 

using Lux 

model = Chain(; bip = f_bip_lux, 
                l1 = Dense(length(f_bip_lux), 10), 
                l2 = Dense(10, 10, tanh), 
                l3 = Dense(10, 1) )

rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, model)              


model(X, ps, st)[1][1] 

@btime $model($X, $ps, $st)

##

data = [ identity.(hyp_jets[i]) for i = 1:100 ]

function loss(model, ps, st, data)
    return sum( model(X, ps, st)[1][1]^2 for X in data )
end

loss(model, ps, st, data)

# standard Lux differentation uses Zygote and more or less goes like this: 
# but somehow this doesn't seem to be working? 
using Zygote 
Zygote.gradient(ps -> loss(model, ps, st, data))

# Enzyme should works much faster though ... 
using Enzyme 
# allocate gradient (Enzyme wants everything non-allocating!)
gs = Lux.fmap(zero, ps)
Enzyme.autodiff(Reverse, loss, Const(model), 
                Duplicated(ps, gs), Const(st), Const(data))
println(gs)
