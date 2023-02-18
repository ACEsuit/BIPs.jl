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
                l1 = Dense(length(f_bip_lux), 10, tanh), 
                l2 = Dense(10, 10, tanh), 
                l3 = Dense(10, 1) )

# for some reason, this chain uses F32 for intermediate operations 
# I don't know how to turn this off. Another reason to write out 
# own layers

rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, model)              

model(X, ps, st)[1][1] 

# @btime $model($X, $ps, $st)



##

using Lux, Optimisers, Zygote

# standard Lux differentation uses Zygote and more or less goes like this: 

data = [ identity.(hyp_jets[i]) for i = 1:100 ]

function loss(model, ps, st, data)
   L = [ model(X, ps, st)[1][1] for X in data ]
   return sum(L.^2), st, () 
end

loss(model, ps, st, data)
print("Time loss: "); 
@time loss(model, ps, st, data)


opt = Optimisers.ADAM(0.001)
train_state = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)

# the timing is actually quite good. Maybe no need to worry about 
# Enzyme for now.
print("Time grad: "); 
@time gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)

# via the same kind of mechanism one can now use the Lux training machinery
#   http://lux.csail.mit.edu/stable/examples/generated/beginner/PolynomialFitting/main/

# If we want to multi-thread this, then we need to deepcopy the parameters 
# and the state. Then we can evaluate the loss and gradient on each thread 
# for a different subset of the training set and in the end combine the 
# gradients. 

## ------------- Enzyme Tests ----------------

# # Unfortunately I can't get the Enzyme differentiation to work. 
# # ... no idea what the problem is?!?
# # ... I suspect if we write out own layers we will be ok?!

# using Enzyme 

# function loss1(model, ps, st, data)
#    L = 0.0  
#    for X in data
#       L += model(X, ps, st)[1][1]^2
#    end
#    return L
# end

# loss1(model, ps, st, data)

# # Enzyme should works much faster though ... 
# #  ... but something isn't working at all here. 
# # allocate gradient (Enzyme wants everything non-allocating!)
# gs = Lux.fmap(zero, ps)
# Enzyme.autodiff(Reverse, loss, Const(loss1), 
#                 Duplicated(ps, gs), Const(st), Const(data))
# println(gs)

##

