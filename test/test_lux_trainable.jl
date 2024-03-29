
using BIPs, Statistics, StaticArrays, Random, Test, ACEcore, Printf,
      Polynomials4ML, LinearAlgebra, LuxCore, Lux, BenchmarkTools
using Polynomials4ML.Testing: print_tf  
using Lux, Optimisers, Zygote

rng = Random.GLOBAL_RNG

include("testing_tools.jl")
X = jets[1]

##

order = 3
maxlevel = 4
n_pt = 3
n_tM = 2
n_th = 2
n_y = 2

tB = BIPs.LuxBIPs.transverse_embedding(; n_pt = n_pt, n_tM = n_tM, maxlen=maxlen)
θB = BIPs.LuxBIPs.angular_embedding(; n_th = n_th, maxlen=maxlen)
yB = BIPs.LuxBIPs.y_embedding(; n_y = n_y, maxlen=maxlen)

f_bip = BIPs.LuxBIPs.bips(tB, θB, yB, order=order, maxlevel=maxlevel)

ps, st = LuxCore.setup(rng, f_bip)
Bs = f_bip(X, ps, st)[1]

##

bip_len = length(f_bip.layers.corr)

model_l = Chain(; bip = f_bip,
                  l1 = Dense(bip_len, 5, tanh; init_weight=randn, use_bias=false), 
                  l2 = Dense(5, 1, tanh; init_weight=randn, use_bias=false),
                  out = WrappedFunction(x -> x[1]), )
        
psl, stl = LuxCore.setup(rng, model_l)

model_l(X, psl, stl)[1]

gl = Zygote.gradient(ps -> model_l(X, ps, stl)[1], psl)[1]

##

fl = ps -> model_l(X, ps, stl)[1]
@btime Zygote.gradient($fl, $psl);


##
# Finite difference test for model_s 

model, ps, st = model_l, psl, stl

ps_vec, re = destructure(ps)
us_vec = ( (randn(length(ps_vec)) ) ./ (1:length(ps_vec)) ) 
         #   .* [ (rand() < 0.1) for _=1:length(ps_vec)] )

_ps(t) = re(ps_vec + t * us_vec)
_dot(nt1::NamedTuple, nt2::NamedTuple) = dot(destructure(nt1)[1], destructure(nt2)[1])

f0 = model(X, ps, st)[1] 
g0 = Zygote.gradient(ps -> model(X, ps, st)[1], ps)[1]
df0 = dot(destructure(g0)[1], us_vec)

for h in (0.1).^(1:12)
   fhp = model(X, _ps(h), st)[1]
   fhm = model(X, _ps(-h), st)[1]
   df_h = (fhp - f0) / h
   df_h2 = (fhp - fhm) / (2*h)
   @printf(" %.2e | %.2e  ,  %.2e \n", h, 
            abs(df_h - df0),
            abs(df_h2 - df0)
             )
end

