using BIPs, Statistics, StaticArrays, Random, Test, ACEcore, 
      Polynomials4ML, LinearAlgebra, LuxCore
using Polynomials4ML.Testing: print_tf      

include("testing_tools.jl")
hyp_jets = sample_hyp_jets
jets = [identity.(jet) for jet in hyp_jets ]

##

order = 3
maxlevel = 6
n_pt = 4
n_th = 2
n_y = 2

f_bip_lux = BIPs.LuxBIPs.simple_bips(; 
            order=order, maxlevel=maxlevel, n_pt=n_pt, n_th=n_th, n_y=n_y)

##

module Ez
   using Lux, LuxCore, Random, LinearAlgebra
   import LuxCore: initialparameters, initialstates, 
                     AbstractExplicitLayer


   struct Dense!{T, TACT} <: AbstractExplicitLayer
      activation::TACT 
      dims::Tuple{Int, Int} # (out, in)
   end

   Dense!(dimin, dimout, activation = identity; T = Float64) = 
         Dense!{T, typeof(activation)}(activation, (dimout, dimin))


   initialparameters(rng::AbstractRNG, l::Dense!{T}) where {T} = 
         (W = T.(Lux.glorot_uniform(rng, l.dims...)), )

   initialstates(rng::AbstractRNG, l::Dense!{T}) where {T} =
         (A = zeros(T, l.dims[2]), B = zeros(T, l.dims[1]), )

   function (l::Dense!)(X, ps, st)
      A, B = st.A, st.B 
      map!(l.activation, A, X) 
      mul!(B, ps.W, A)
      return B, st 
   end

end

##

using Lux 

model = Chain(; bip = f_bip_lux, 
                l1 = Ez.Dense!(length(f_bip_lux), 10, tanh), 
                l2 = Ez.Dense!(10, 10, tanh), 
                l3 = Ez.Dense!(10, 1), 
                out = WrappedFunction(x -> x[1]), )

# for some reason, this chain uses F32 for intermediate operations 
# I don't know how to turn this off. Another reason to write out 
# own layers

rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, model)              

X = jets[1]
model(X, ps, st)[1]

##
using BenchmarkTools
@btime $model($X, $ps, $st)

## ------------- Enzyme Tests ----------------

using Enzyme 

function loss1(model, ps, st, data)
   L = 0.0  
   for X in data
      L += model(X, ps, st)[1]^2
   end
   return L / length(data)
end

loss1(model, ps, st, jets)

# allocate gradient (Enzyme wants everything non-allocating!)
gs = Lux.fmap(zero, ps)
Enzyme.autodiff(Reverse, loss1, Const(model),
                Duplicated(ps, gs), Const(st), Const(jets))
println(gs)

##

