
using ChainRules: ignore_derivatives
using Random 

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


# ----------------- a basic learnable embedding 
#  ... of the form 
#   Rk = ∑_k' W_{kk'} R0_{k'}

using LuxCore: AbstractExplicitContainerLayer
using Tullio: @tullio
using Lux: glorot_normal

struct RtMEmbedding{T, TR, TM} <: AbstractExplicitContainerLayer{(:r_embed, :m_embed)}
   r_embed::TR
   m_embed::TM
   nmax::Int
   maxlen::Int 
   meta::Dict{String, Any}
end

RtMEmbedding(T, r_embed::TR, m_embed::TM, nmax::Integer; 
             maxlen=200)  where {TR, TM} =  
      RtMEmbedding{T, TR, TM}(r_embed, m_embed, nmax, maxlen, Dict{String, Any}())

function initialparameters(rng::AbstractRNG, l::RtMEmbedding{T}) where {T} 
   dims = (l.nmax, length(l.r_embed), length(l.m_embed))
   W = T.( glorot_normal(rng, dims...) )
   return ( W=W, 
            r_embed = initialparameters(l.r_embed),
            m_embed = initialparameters(l.m_embed) )
end

initialparameters(l::RtMEmbedding)  = 
      initialparameters(Random.GLOBAL_RNG, l) 
       
initialstates(rng::AbstractRNG, l::RtMEmbedding) = 
                  ( r_embed = initialstates(rng, l.r_embed),
                    m_embed = initialstates(rng, l.m_embed) )

initialstates(l::RtMEmbedding) = initialstates(Random.GLOBAL_RNG, l)

function (l::RtMEmbedding{T})(X::AbstractVector{<: SVector}, ps, st) where {T} 
   nX = length(X)
   @assert nX <= l.maxlen
   R, _ = l.r_embed(X, ps.r_embed, st.r_embed)
   R_ = @view R[1:nX, :]
   M, _ = l.m_embed(X, ps.m_embed, st.m_embed)
   M_ = @view M[1:nX, :]
   P, _ = Matrix{T}(undef, length(X), l.nmax)
   @tullio P[i, n] := ps.W[n, k1, k2] * R_[i, k1] * M_[i, k2]
   # here we can release R, M
   return P, st 
end

## ---------------- Varaint of Lux.Bilinear suitable for ACE

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
   inds_pt = natural_indices(cheb)
   inv_pt = idx_map(cheb)
   bR = ConstEmbedding(T, T, pt_trans, cheb, maxlen)

   # tM embedding 
   mono = Polynomials4ML.MonoBasis(n_tM) 
   inds_tM = natural_indices(mono)
   inv_tM = idx_map(mono)
   bM = ConstEmbedding(T, T, tM_trans, mono, maxlen)
      
   bT = RtMEmbedding(T, bR, bM, nmax, maxlen=maxlen)
   bT.meta["inds"] = 0:nmax-1 
   bT.meta["inv"] = Dict([i => i+1 for i = 0:nmax-1]...)

   return bT
end

function transverse_embedding2(; 
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
