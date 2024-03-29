
using ChainRules: ignore_derivatives
using Random 
using Tullio
using Lux: glorot_normal

# --------------  
# some useful auxiliary functions 

function inv_collection(a)
   ia = Dict{eltype(a), Int}()
   for (i, ai) in enumerate(a)
      ia[ai] = i 
   end
   return ia 
end

function idx_map(basis)
   a = natural_indices(basis)
   ia = Dict{eltype(a), Int}() 
   for ai in a
      ia[ai] = Polynomials4ML.index(basis, ai)
   end
   return ia 
end


# ----- 
# the standard chebyshev basis, this should probably 
# go into Polynomials4ML.jl

function simple_chebyshev(maxn)
   cheb = chebyshev_basis(maxn+1)
   cheb.A[1:2] .= 1.0 
   cheb.A[3:end] .= 2 
   cheb.B[:] .= 0.0 
   cheb.C[:] .= -1.0 
   return cheb
end


# ----------- a simple wrapper that just says the layers inside not trainable 
#

"""
a simple wrapper that just says the layers inside not trainable 
just wraps the evaluation of the inner layer in a ignore_derivative() 
this gives undefined results if the parameters are anything other than 
empty NamedTuples
"""
struct ConstL{TL} <: AbstractExplicitLayer
   l::TL
end

Base.length(l::ConstL) = length(l.l)

initialparameters(rng::AbstractRNG, l::ConstL) = 
      (l = initialparameters(rng, l.l), )

initialstates(rng::AbstractRNG, l::ConstL) = 
      (l = initialstates(rng, l.l), )

function (l::ConstL)(x, ps, st) 
   return ignore_derivatives() do 
      l.l(x, ps.l, st.l)
   end
end

(l::ConstL)(x) = 
      l(x, initialparameters(Random.GLOBAL_RNG, l), 
           initialstates(Random.GLOBAL_RNG, l) )[1]


# ----------- a simple embedding interface 

"""
Lux wrapper for a Polynomials4ML embedding. Allows no parameters. 
Adds a transform in front. Efficient batched evaluation. Neither transform 
nor embedding are allowed to be trainable.
"""
struct ConstEmbedding{TIN, TOUT, TT, TB} <: AbstractExplicitLayer 
   transform::TT
   B::TB
   maxlen::Int 
   meta::Dict{String, Any}
end

ConstEmbedding(TIN, TOUT, transform, B, maxlen) = 
      ConstEmbedding{TIN, TOUT, typeof(transform), typeof(B)}(
         transform, B, maxlen, Dict{String, Any}()
      )


Base.length(l::ConstEmbedding) = length(l.B)

function (l::ConstEmbedding)(X, ps, st) 
   P = ignore_derivatives() do
      _eval(l::ConstEmbedding, X, ps, st)
   end
   return P, st 
end

function _eval(l::ConstEmbedding, X, ps, st) 
   nX = length(X) 
   x = @view st.x[1:nX]
   P = @view st.P[1:nX, :]
   @assert length(x) >= nX
   @assert size(P, 1) >= nX

   # transform input to correct format 
   @simd ivdep for i = 1:nX 
      @inbounds x[i] = l.transform(X[i])
   end
   # now evaluate the embedding 
   Polynomials4ML.evaluate!(P, l.B, x[1:nX])
   return P 
end

initialparameters(rng::AbstractRNG, l::ConstEmbedding) = 
      initialparameters(l) 

initialparameters(l::ConstEmbedding) = 
         NamedTuple() 

initialstates(rng::AbstractRNG, l::ConstEmbedding) = 
         initialstates(l) 

initialstates(l::ConstEmbedding{TIN, TOUT}) where {TIN, TOUT} = (
         x = Vector{TIN}(undef, l.maxlen), 
         P = Matrix{TOUT}(undef, l.maxlen, length(l.B))
      )
         

# ---------------- BIP Radial embedding Layer - simple version 

# this assumes the r_trans and r_embed are not trainable
struct SimpleRtMEmbedding{T, TTR, TR} <: AbstractExplicitLayer
   r_trans::TTR
   r_embed::TR
   maxlen::Int 
   meta::Dict{String, Any}
end

Base.length(l::SimpleRtMEmbedding) = length(l.r_embed)

SimpleRtMEmbedding(T, r_trans, r_embed, maxlen) = 
      SimpleRtMEmbedding{T, typeof(r_trans), typeof(r_embed)}(r_trans, r_embed, 
                         maxlen, Dict{String, Any}())

initialparameters(rng::AbstractRNG, l::SimpleRtMEmbedding) = 
      initialparameters(l) 

initialparameters(l::SimpleRtMEmbedding) = 
      NamedTuple()
         
initialstates(rng::AbstractRNG, l::SimpleRtMEmbedding) = 
         initialstates(l) 

initialstates(l::SimpleRtMEmbedding{T}) where {T} = (
      r = Vector{T}(undef, l.maxlen), 
      tM = Vector{T}(undef, l.maxlen), 
      R = Matrix{T}(undef, l.maxlen, length(l.r_embed))
   )

function (l::SimpleRtMEmbedding)(X, ps, st)
   R = ignore_derivatives() do 
      _eval(l, X, ps, st)
   end
   return R, st 
end 

function _eval(l::SimpleRtMEmbedding, X, ps, st)
   nX = length(X) 
   r = @view st.r[1:nX]
   tM = st.tM 
   R = @view st.R[1:nX, :]
   @assert size(R, 1) >= nX 
   @assert length(r) >= nX 
   @assert length(tM) >= nX 
   @assert size(R, 2) >= length(l.r_embed)

   @inbounds begin 
      @simd ivdep for i = 1:nX 
         x = X[i] 
         # (log(x[1]) + 4.7) / 6   # log(x[1] + a) / b 
         r[i] = l.r_trans(x)
         tM[i] = x[5] 
      end

      # evaluate the r embedding 
      Polynomials4ML.evaluate!(R, l.r_embed, r)

      # apply the tM rescaling 
      @inbounds for j = 1:length(l.r_embed)
         @simd ivdep for i = 1:nX
            R[i, j] *= tM[i]
         end
      end

   end

   return R 
end


## ---------------- Varaint of Lux.Bilinear suitable for ACE
# needed for the embedding of pt and tM 
struct BatchedBilinear <: AbstractExplicitLayer
   nin1::Int 
   nin2::Int 
   nout::Int
   meta::Dict{String, Any}
end

function BatchedBilinear(dims::Pair{<: Tuple, <: Integer})
   nin = dims[1] 
   nout = dims[2]
   @assert length(nin) == 2
   nin1, nin2 = nin  
   return BatchedBilinear(nin1, nin2, nout, Dict{String, Any}())
end


function initialparameters(rng::AbstractRNG, l::BatchedBilinear)
   dims = (l.nout, l.nin1, l.nin2)
   W = Float64.( glorot_normal(rng, dims...) )
   return ( W=W,  )
end
       
initialstates(rng::AbstractRNG, l::BatchedBilinear) = NamedTuple() 

(l::BatchedBilinear)(RM::Tuple, ps, st) = _eval(l, RM, ps.W, st), st 

function _eval(l::BatchedBilinear, RM::Tuple, W, st)
   @assert length(RM) == 2 
   R, M = RM 
   @tullio P[i, n] := W[n, k1, k2] * R[i, k1] * M[i, k2]
   return P
end

# we could bring in this rrule. But no rush 
# let's first benchmark properly and find the bottlenecks 
# function rrule(::typeof(_eval), l::BatchedBilinear, RM::Tuple, W, st)
#    P = _eval(l, RM, W, st)

#    function _eval_pullback(ΔP)
#       @tullio ΔR[i, k1] := W[n, k1, k2] * M[i, k2] * ΔP[i, n]
#       @tullio ΔM[i, k2] := W[n, k1, k2] * R[i, k1] * ΔP[i, n]
#       return NoTangent(), NoTangent(), (ΔR, ΔM), NoTangent()
#    end

#    return P, _eval_pullback
# end



# ------------------ convenience constructors 
# for various embeddings 


function angular_embedding(; n_th = 2, maxlen = 200)
   # angular embedding 
   trig_θ = CTrigBasis(n_th)
   bT = ConstEmbedding(Float64, ComplexF64, 
                          x -> atan(x[3], x[2]), trig_θ, maxlen)
   bT.meta["info"] = "complex trig embedding of angle θ" 
   bT.meta["inds"] = natural_indices(trig_θ)
   bT.meta["inv"] = idx_map(trig_θ)
   return bT
end

function y_embedding(; n_y = 2, maxlen = 200)
   trig_y = CTrigBasis(n_y)
   bY = ConstEmbedding(Float64, ComplexF64, 
                          x -> x[4], trig_y, maxlen)
   bY.meta["info"] = "complex trig embedding of transverse momentum y" 
   bY.meta["inds"] = natural_indices(trig_y)
   bY.meta["inv"] = idx_map(trig_y)
   return bY                             
end

function simple_transverse_embedding(; n_pt = 5, maxlen = 200, 
                                    pt_trans = x -> (log(x[1]) + 4.7) / 6)
   cheb = simple_chebyshev(n_pt)
   bR = SimpleRtMEmbedding(Float64, pt_trans, cheb, maxlen)
   bR.meta["inds"] = natural_indices(cheb)
   bR.meta["inv"] = idx_map(cheb)
   bR.meta["info"] = "basic original BIPs transverse momentum embedding"
   return  bR 
end 

function transverse_embedding(; 
                        pt_trans = x -> (log(x[1]) + 4.7) / 6,
                        n_pt = 5, 
                        tM_trans = x -> x[5], 
                        n_tM = 2, 
                        nmax = n_pt, 
                        T = Float64, 
                        maxlen = 200 )
   # pt embedding 
   cheb = simple_chebyshev(n_pt)
   bR = ConstEmbedding(T, T, pt_trans, cheb, maxlen)
   # tM embedding 
   mono = Polynomials4ML.MonoBasis(n_tM) 
   bM = ConstEmbedding(T, T, tM_trans, mono, maxlen)

   # create a bilinear learnable map
   len_r = length(cheb)
   len_m = length(mono)
   bR_l = Chain( 
         embeddings = BranchLayer((r = bR, m = bM)), 
         bilinear = BIPs.LuxBIPs.BatchedBilinear((len_r, len_m) => nmax), 
         ) |> BIPs.LuxBIPs.MetaLayer
   bR_l.meta["inds"] = 0:nmax-1
   bR_l.meta["inv"] = Dict([i => i+1 for i = 0:nmax-1]...)
   return bR_l 
end
