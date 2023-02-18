
using BIPs, Statistics, StaticArrays, Random, LinearAlgebra, Lux, 
       Optimisers

include("../../test/testing_tools.jl")
hyp_jets = sample_hyp_jets
# make them type-stable 
jets = [ identity.(X) for X in hyp_jets ]
labels = identity.(sample_labels)

# need to initialize a rng 
rng = MersenneTwister(1234)

##

f_bip, specs = build_ip(order=3,
      levels=6,
      n_pt=4,
      n_th=2,
      n_y=2)

f_bip_lux = BIPs.LuxBIPs.BIPbasis(f_bip)

model = Chain(; bip = f_bip_lux, 
                hidden1 = Dense(length(f_bip_lux), 10, tanh), 
                hidden2 = Dense(10, 10, tanh), 
                readout = Dense(10, 1, tanh), 
                tonum = WrappedFunction(x -> (1+x[1])/2) )

ps, st = Lux.setup(rng, model)              
model(jets[1], ps, st)[1]

##
# serial implementation of the training loop 

data = (jets, labels)

function loss_function(model, ps, st, data)
   P = [ model(jet, ps, st)[1] for jet in data[1] ]
   Y = data[2]
   return sum( Y .* log.(P) + (1 .- Y) .* log.(1 .- P) ), st, ()
end

# loss(model, ps, st, data)


function serial(tstate, vjp, data, epochs)
   for epoch in 1:epochs
      grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function, data, tstate)
      @info epoch=epoch loss=loss
      tstate = Lux.Training.apply_gradients(tstate, grads)
   end
   return tstate
end

opt = Optimisers.ADAM(0.001)
train_state = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

# take 10 training steps
tstate = serial(train_state, vjp, data, 10)

# trained parameters:
ps_opt = tstate.parameters


## 
# parallel implementation of the training loop



