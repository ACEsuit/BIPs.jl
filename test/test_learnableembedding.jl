using BIPs, Statistics, StaticArrays, Random, Test, ACEcore, 
      Polynomials4ML, LinearAlgebra, LuxCore, Lux, BenchmarkTools
using Polynomials4ML.Testing: print_tf  

rng = MersenneTwister(1234)

include("testing_tools.jl")
hyp_jets = sample_hyp_jets
jets = [ identity.(jet) for jet in hyp_jets ]
maxlen = maximum(length(jet) for jet in jets)

##

order = 3
maxlevel = 6
n_pt = 4
n_tM = 2
n_th = 2
n_y = 2

tB = BIPs.LuxBIPs.transverse_embedding(; n_pt = n_pt, n_tM = n_tM, maxlen=maxlen)
θB = BIPs.LuxBIPs.angular_embedding(; n_th = n_th, maxlen=maxlen)
yB = BIPs.LuxBIPs.y_embedding(; n_y = n_y, maxlen=maxlen)

f_bip = BIPs.LuxBIPs.bips(tB, θB, yB, order=order, maxlevel=maxlevel)

X = jets[1]
ps, st = LuxCore.setup(rng, f_bip)
f_bip(X, ps, st)[1]

##

model = Chain(; bip = f_bip, 
                l1 = Dense(length(f_bip), 10, tanh), 
                l2 = Dense(10, 10, tanh), 
                l3 = Dense(10, 1), 
                out = WrappedFunction(x -> x[1]), )

ps, st = Lux.setup(rng, model)              

model(X, ps, st)[1]

@btime $model($X, $ps, $st)

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


opt = Optimisers.ADAM(0.001)
train_state = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)

## the timing is actually quite decent
print("Time grad: "); 
@time gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)

##
# Finite difference test 

using Printf 

ps_vec, re = destructure(ps)
us_vec = randn(length(ps_vec)) ./ (1:length(ps_vec)).^2
_ps(t) = re(ps_vec + t * us_vec)
_dot(nt1::NamedTuple, nt2::NamedTuple) = dot(destructure(nt1)[1], destructure(nt2)[1])

f0 = loss(model, ps, st, data)[1] 
df0 = dot(destructure(gs)[1], us_vec)

for h in (0.1).^(2:10)
   fh = loss(model, _ps(h), st, data)[1]
   df_h = (fh - f) / h
   @printf(" %.2e | %.2e \n", h, abs(df_h - df0) )
end