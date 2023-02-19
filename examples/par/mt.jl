
using BIPs, Statistics, StaticArrays, Random, LinearAlgebra, Lux, 
       Optimisers, ThreadsX

# example of serial and multi-threaded training loop 
# not very elegant but the main functionality I'd like to use 
# doesn't appear to support Zygote differentiation?!?       
# there is also a ridiculous amount of memory allocation going on
# that I want to track down.

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
   jets, Y = data
   P = map(jet -> model(jet, ps, st)[1], jets)
   return sum(Y .* log.(P) .+ (1 .- Y) .* log.(1 .- P)), st, ()
end

loss_function(model, ps, st, data)

function main(tstate, vjp, data, epochs)
   for epoch in 1:epochs
      grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function, data, tstate)
      (mod(epoch, 10) == 0) && (@info epoch=epoch loss=loss)
      tstate = Lux.Training.apply_gradients(tstate, grads)
   end
   return tstate
end

opt = Optimisers.ADAM(0.001)
train_state = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

# take 10 training steps
@time tstate = main(train_state, vjp, data, 100)

# trained parameters:
ps_opt = tstate.parameters


## 
# multi-threaded implementation of the training loop
# the problem is that if we do it naively then Zygote complains 
# One would think that Folds.map or ThreadsX.map would work but 
# they don't have rrules, so unfortunately we need to manage this
# manually. :(

function split_data(data::Tuple, N::Integer) 
   Ndat = length(data[1])
   Nblock = ceil(Int, Ndat / N)
   jets = [ data[1][i:min(i+Nblock-1, Ndat)] for i in 1:Nblock:Ndat ]
   labels = [ data[2][i:min(i+Nblock-1, Ndat)] for i in 1:Nblock:Ndat ]
   return [ (jets[i], labels[i]) for i in 1:length(jets) ]
end


function mt_gradients(vjp, loss_function, mt_data, tstate)
   nt = length(mt_data)
   mt_tst = [ deepcopy(tstate) for i in 1:nt ]
   mt_out = ThreadsX.map(1:nt) do i
      # grads, loss, stats, tstate
      Lux.Training.compute_gradients(vjp, loss_function, mt_data[i], mt_tst[i])
   end
   _add(x, y) = x + y 
   _add(::Nothing, ::Nothing) = nothing              
   loss = sum(mt_out[i][2] for i in 1:nt)
   grads = mt_out[1][1] 
   for i in 2:nt
      grads = Lux.fmap(_add, grads, mt_out[i][1])
   end
   return grads, loss
end

function mt_main(tstate, vjp, data, epochs)
   data_split = split_data(data, Threads.nthreads())
   for epoch in 1:epochs
      grads, loss = mt_gradients(vjp, loss_function, data_split, tstate)
      (mod(epoch, 10) == 0) && @info epoch=epoch loss=loss
      tstate = Lux.Training.apply_gradients(tstate, grads)
   end
   return tstate
end

opt = Optimisers.ADAM(0.001)
tstate = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.ZygoteVJP()

# take 10 training steps
@time tstate = mt_main(tstate, vjp, data, 100)

# trained parameters:
ps_opt = tstate.parameters
