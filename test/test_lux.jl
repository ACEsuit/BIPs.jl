using BIPs, Statistics, StaticArrays, Random, Test, ACEcore, 
      Polynomials4ML, LinearAlgebra, LuxCore
using Polynomials4ML.Testing: print_tf      

include("testing_tools.jl")

##

@info("Testing Compatibility of Lux Bips with Original BIPs")

order = 3
maxlevel = 6
n_pt = 4
n_th = 2
n_y = 2

f_bip, specs = build_ip(; 
            order=order, levels=maxlevel, n_pt=n_pt, n_th=n_th, n_y=n_y)

f_bip_lux = BIPs.LuxBIPs.simple_bips(; 
            order=order, maxlevel=maxlevel, n_pt=n_pt, n_th=n_th, n_y=n_y)

##

function _f_bip_lux(X) 
   ps, st = LuxCore.setup(MersenneTwister(1234), f_bip_lux)
   f_bip_lux(X, ps, st)[1]
end

@info("old vs new implementation")
for X in jets[1:100]
   AA1 = f_bip(X)
   AA2 = _f_bip_lux(X)
   print_tf(@test norm((AA1 - AA2) ./ abs.(AA2), Inf) < 0.1)
   print_tf(@test norm(AA1 - AA2, Inf) / norm(AA2, Inf) < 1e-4)
end
println() 

##

@info("checkout the human-readable spec")
spec_A = BIPs.LuxBIPs.get_1p_spec(f_bip_lux)
spec_AA = BIPs.LuxBIPs.get_bip_spec(f_bip_lux)
display(spec_AA[1:10]) 
println("...")
display(spec_AA[end-5:end])

##

using  BenchmarkTools, LuxCore
X = jets[1] 
@info("Original")
@btime $f_bip($X)

@info("Lux style")
# benchmark when we pre-allocate
rng = MersenneTwister(1234)
ps, st = LuxCore.setup(rng, f_bip_lux)
@btime $f_bip_lux($X, $ps, $st)

##

using Lux 

len_bip = length(f_bip_lux.layers.corr)
model = Chain(; bip = f_bip_lux, 
                l1 = Dense(len_bip, 10, tanh), 
                l2 = Dense(10, 10, tanh), 
                l3 = Dense(10, 1), 
                out = WrappedFunction(x -> x[1]), )

# for some reason, this chain uses F32 for intermediate operations 
# there seems no obvious way to switch for F64 expcept to 
# specify randn for the initialisation of weights - to be discussed?!

rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, model)              

model(X, ps, st)[1]

# @btime $model($X, $ps, $st)



##

using Lux, Optimisers, Zygote

# standard Lux differentation uses Zygote and more or less goes like this: 

data = jets[1:100]

function loss(model, ps, st, data)
   L = [ model(X, ps, st)[1][1] for X in data ]
   return sum(L.^2), st, () 
end

loss(model, ps, st, data)
print("Time loss: "); 
@time loss(model, ps, st, data)
@time loss(model, ps, st, data)

opt = Optimisers.Adam(0.001)
train_state = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)

# the timing is actually quite good. Maybe no need to worry about 
# Enzyme for now.
print("Time grad: "); 
@time gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)
@time gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)

# via the same kind of mechanism one can now use the Lux training machinery
# cf. examples/lux 

