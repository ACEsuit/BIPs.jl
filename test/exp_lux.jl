using Lux, LuxCore, Printf, Random, Optimisers, LinearAlgebra, 
       ChainRules, ChainRulesCore
rng = Random.GLOBAL_RNG
## ----------- a stupid toy model 

struct MyPool <: LuxCore.AbstractExplicitContainerLayer{(:l1,)}
   nmax::Int
   l1
end

LuxCore.initialparameters(rng::AbstractRNG, l::MyPool) = (l1 = NamedTuple(),)

LuxCore.initialstates(rng::AbstractRNG, l::MyPool) = (l1 = NamedTuple(),)

(l::MyPool)(X::AbstractVector, ps, st) = _apply(l, X, ps, st), st

_apply(l::MyPool, X, ps, st) = [ sum(X.^n) for n = 1:l.nmax ]


function ChainRulesCore.rrule(::typeof(_apply), l::MyPool, X, ps, st)
   P = _apply(l, X, ps, st)

   function _apply_pullback(Δ)
      @show Δ
      return NoTangent(), NoTangent(), NoTangent(), ZeroTangent(), NoTangent() 
   end
   return P, _apply_pullback
end


##


data  = [ randn(10) for _=1:10 ] 

function loss(model, ps, st, data)
   L = [ model(X, ps, st)[1] for X in data ]
   return sum(L.^2), st, () 
end

model = Chain(; pool = MyPool(5, nothing),
                l1 = Dense(5, 10, tanh; init_weight=randn), 
                l2 = Dense(10, 10, tanh; init_weight=randn), 
                l3 = Dense(10, 1; init_weight=randn), 
                out = WrappedFunction(x -> x[1]), )

ps, st = Lux.setup(rng, model)              

loss(model, ps, st, data)

opt = Optimisers.ADAM(0.001)
train_state = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

gs, l, _, tst = Lux.Training.compute_gradients(vjp, loss, data, train_state)
ps = tst.parameters
st = tst.states
loss(model, ps, st, data)[1] ≈ l

ps_vec, re = destructure(ps)
us_vec = randn(length(ps_vec)) ./ (1:length(ps_vec))
_ps(t) = re(ps_vec + t * us_vec)
_dot(nt1::NamedTuple, nt2::NamedTuple) = dot(destructure(nt1)[1], destructure(nt2)[1])

f0 = loss(model, ps, st, data)[1] 
f0 ≈ l 
df0 = dot(destructure(gs)[1], us_vec)

for h in (0.1).^(2:10)
   fhp = loss(model, _ps(h), st, data)[1]
   fhm = loss(model, _ps(-h), st, data)[1]
   df_h = (fhp - f0) / h
   @printf(" %.2e | %.2e \n", h, abs(df_h - df0) )
end

##

