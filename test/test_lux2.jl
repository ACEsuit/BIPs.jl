using BIPs, Statistics, StaticArrays, Random, Test, ACEcore, Printf,
      Polynomials4ML, LinearAlgebra, LuxCore, Lux, BenchmarkTools
using Polynomials4ML.Testing: print_tf  
using Lux, Optimisers, Zygote

rng = MersenneTwister(1234)

include("testing_tools.jl")
hyp_jets = sample_hyp_jets
jets = [ identity.(jet) for jet in hyp_jets ]
maxlen = maximum(length, jets)
X = jets[1]

##

@info("Testing Gradient of simple_bips")

order = 3
maxlevel = 3
n_pt = 3
n_tM = 2
n_th = 2
n_y = 2

f_bip_s = BIPs.LuxBIPs.simple_bips2(; 
            order=order, maxlevel=maxlevel, 
            n_pt=n_pt, n_th=n_th, n_y=n_y, 
            maxlen = maxlen)

pss, sts = LuxCore.setup(rng, f_bip_s)
Bs = f_bip_s(X, pss, sts)[1]

len_f_bip = length(f_bip_s.layers.corr)

##

model = Chain(; bip = f_bip_s, 
                norm = WrappedFunction(x -> x/1e4),
                l1 = Dense(len_f_bip, 5, tanh; init_weight=randn, use_bias=false), 
                l2 = Dense(5, 1, tanh; init_weight=randn, use_bias=false),
                out = WrappedFunction(x -> x[1]), 
                )

ps, st = Lux.setup(rng, model)     

[ model(X, ps, st)[1] for X in jets[1:3] ]


##

data = jets[1:3]

function loss(model, ps, st, data)
   L = [ model(X, ps, st)[1] for X in data ]
   return sum(L.^2), st, () 
end

loss(model, ps, st, data)

opt = Optimisers.ADAM(0.001)
train_state = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

gs, l, _, ts = Lux.Training.compute_gradients(vjp, loss, data, train_state)


##
# Finite difference test 

gs, l, _, tst = Lux.Training.compute_gradients(vjp, loss, data, train_state)
ps = tst.parameters
st = tst.states
loss(model, ps, st, data)[1] ≈ l

ps_vec, re = destructure(ps)
us_vec = ( (randn(length(ps_vec)) ) # ./ (1:length(ps_vec))) 
           .* [ (rand() < 0.1) for _=1:length(ps_vec)] )

_ps(t) = re(ps_vec + t * us_vec)
_dot(nt1::NamedTuple, nt2::NamedTuple) = dot(destructure(nt1)[1], destructure(nt2)[1])

##

f0 = loss(model, ps, st, data)[1] 
f0 ≈ l 
df0 = dot(destructure(gs)[1], us_vec)

for h in (0.1).^(1:12)
   fhp = loss(model, _ps(h), st, data)[1]
   fhm = loss(model, _ps(-h), st, data)[1]
   df_h = (fhp - f0) / h
   df_h2 = (fhp - fhm) / (2*h)
   @printf(" %.2e | %.2e  ,  %.2e \n", h, 
            abs(df_h - df0),
            abs(df_h2 - df0)
             )
end

