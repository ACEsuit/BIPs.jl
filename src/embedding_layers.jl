
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
      (l = initialparameters(l.l), ) 

initialstates(rng::AbstractRNG, l::ConstL) = 
      (l = initialstates(rng, l.l), )

function (l::ConstL)(x, ps, st) 
   return ignore_derivatives() do 
      l.l(x, ps.l, st.l)
   end
end


# ----------- a simple embedding interface 
#             so that We can make learnable embeddings; cf below 

# careful, this assumes that transform and B are !not! trainable 
# This Lux wrapper should go into ACEbase or Polynomials4ML 
# but for now this is a convenient prototype implementation
struct GenericEmbedding{TIN, TOUT, TT, TB} <: AbstractExplicitLayer 
   transform::TT
   B::TB
   maxlen::Int 
   meta::Dict{String, Any}
end

GenericEmbedding(TIN, TOUT, transform, B, maxlen) = 
      GenericEmbedding{TIN, TOUT, typeof(transform), typeof(B)}(
         transform, B, maxlen, Dict{String, Any}()
      )

Base.length(l::GenericEmbedding) = length(l.B)

function (l::GenericEmbedding)(X, ps, st) 
   P = ignore_derivatives() do
      _eval(l::GenericEmbedding, X, ps, st)
   end
   return P, st 
end

function _eval(l::GenericEmbedding, X, ps, st) 
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

initialparameters(rng::AbstractRNG, l::GenericEmbedding) = 
      initialparameters(l) 

initialparameters(l::GenericEmbedding) = 
         NamedTuple() 

initialstates(rng::AbstractRNG, l::GenericEmbedding) = 
         initialstates(l) 

initialstates(l::GenericEmbedding{TIN, TOUT}) where {TIN, TOUT} = (
         x = Vector{TIN}(undef, l.maxlen), 
         P = Matrix{TOUT}(undef, l.maxlen, length(l.B))
      )
         

# ---------------- BIP Radial embedding Layer - simple version 

# this assumes the r_trans and r_embed are not trainable
struct SimpleRtMEmbedding{T, TTR, TR} <: AbstractExplicitLayer
   r_trans::TTR
   r_embed::TR
   maxlen::Int 
end

Base.length(l::SimpleRtMEmbedding) = length(l.r_embed)

SimpleRtMEmbedding(T, r_trans, r_embed, maxlen) = 
      SimpleRtMEmbedding{T, typeof(r_trans), typeof(r_embed)}(r_trans, r_embed, maxlen)

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


# ------------------ convenience constructors


function angular_embedding(; n_th = 2, maxlen = 200)
   # angular embedding 
   trig_θ = CTrigBasis(n_th)
   inds_θ = natural_indices(trig_θ)
   inv_θ = idx_map(trig_θ)
   bT = GenericEmbedding(Float64, ComplexF64, 
                          x -> atan(x[3], x[2]), trig_θ, maxlen)
   bT.meta["inds"] = inds_θ
   bT.meta["inv"] = inv_θ
   return bT
end

function y_embedding(; n_y = 2, maxlen = 200)
   trig_y = CTrigBasis(n_y)
   inds_y = natural_indices(trig_y)
   inv_y = idx_map(trig_y)
   bY = GenericEmbedding(Float64, ComplexF64, 
                          x -> x[4], trig_y, maxlen)
   bY.meta["inds"] = inds_y
   bY.meta["inv"] = inv_y
   return bY                             
end

function transverse_embedding(; pt_trans = x -> (log(x[1]) + 4.7) / 6,
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
   bR = GenericEmbedding(T, T, pt_trans, cheb, maxlen)

   # tM embedding 
   mono = Polynomials4ML.MonoBasis(n_tM) 
   inds_tM = natural_indices(mono)
   inv_tM = idx_map(mono)
   bM = GenericEmbedding(T, T, tM_trans, mono, maxlen)
      
   bT = RtMEmbedding(T, bR, bM, nmax, maxlen=maxlen)
   bT.meta["inds"] = 0:nmax-1 
   bT.meta["inv"] = Dict([i => i+1 for i = 0:nmax-1]...)

   return bT
end

