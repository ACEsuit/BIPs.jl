using Distributed 
addprocs(8)   

##

# this seems to yield an error no matter what I do but 
# it still works --- no idea what's going on here? 
@everywhere begin 
   using Pkg 
   Pkg.activate(@__DIR__()); Pkg.instantiate()

   using BIPs, Statistics, StaticArrays, Random, LinearAlgebra, Lux, 
         Optimisers, ParallelDataTransfer
end 

##


@everywhere begin 
   include("../../test/testing_tools.jl")
   hyp_jets = sample_hyp_jets
   # make them type-stable 
   jets = [ identity.(X) for X in hyp_jets ]
   labels = identity.(sample_labels)
   maxlen = maximum(length, jets)

   # need to initialize a rng 
   rng = MersenneTwister(1234)
   n_pt = 4; n_tM = 2; n_th = 2; n_y = 2; order = 3; maxlevel = 6;
   nmax = 4

   # our old standard BIP embedding 
   # f_bip = BIPs.LuxBIPs.simple_bips(; order=order, maxlevel=maxlevel, 
   #                                    n_pt=n_pt, n_th=n_th, n_y=n_y)

   # or with learnable transverse embedding: 
   tB = BIPs.LuxBIPs.transverse_embedding(; n_pt = n_pt, n_tM = n_tM, nmax=nmax, maxlen=maxlen)
   θB = BIPs.LuxBIPs.angular_embedding(; n_th = n_th, maxlen=maxlen)
   yB = BIPs.LuxBIPs.y_embedding(; n_y = n_y, maxlen=maxlen)
   f_bip = BIPs.LuxBIPs.bips(tB, θB, yB, order=order, maxlevel=maxlevel)

   len_bip = length(f_bip.layers.corr)

   model = Chain(; bip = f_bip, 
                  hidden1 = Dense(len_bip, 10, tanh), 
                  hidden2 = Dense(10, 10, tanh), 
                  readout = Dense(10, 1, tanh), 
                  tonum = WrappedFunction(x -> (1+x[1])/2) )

   ps, st = Lux.setup(rng, model)
   model(jets[1], ps, st)[1]
end

##
# fix subsets of data on each worker 

function split_data(Ndat::Integer, Nproc::Integer)
   Nblock = ceil(Int, Ndat / Nproc)
   return [ i:min(i+Nblock-1, Ndat) for i in 1:Nblock:Ndat ]
end

subsets = split_data(length(jets), nprocs())

for (i, iproc) in enumerate(procs())
   sendto(iproc, data_subset = subsets[i])
end

@everywhere begin 
   jets = [ identity.(X) for X in hyp_jets[data_subset] ]
   labels = identity.(sample_labels[data_subset])
   data = (jets, labels)
end

##


@everywhere begin 

   function loss_function(model, ps, st, data)
      jets, Y = data
      P = map(jet -> model(jet, ps, st)[1], jets)
      return sum(Y .* log.(P) .+ (1 .- Y) .* log.(1 .- P)), st, ()
   end

   function grad_loss(data, tstate)
      vjp = Lux.Training.ZygoteVJP()
      grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function, data, tstate)
      return grads, loss
   end

end 


function dist_main(tstate, epochs)
   for epoch in 1:epochs
      sendto(procs(), tstate = tstate)
      @everywhere begin 
         grads, loss = grad_loss(data, tstate)
      end
      _grad = grads 
      _loss = loss
      _add(x, y) = x + y 
      _add(::Nothing, ::Nothing) = nothing              
      for iproc in workers()
         _loss += getfrom(iproc, :loss)
         _grad = Lux.fmap(_add, _grad, getfrom(iproc, :grads))
      end
      (mod(epoch, 10) == 0) && (@info epoch=epoch loss=_loss)
      tstate = Lux.Training.apply_gradients(tstate, _grad)
   end
   return tstate
end

##

opt = Optimisers.ADAM(0.001)
tstate = Lux.Training.TrainState(rng, model, opt)

# take 10 training steps
@time tstate = dist_main(tstate, 100)

# trained parameters:
ps_opt = tstate.parameters

##
