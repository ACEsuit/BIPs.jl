

# ----------- a simple embedding interface 
#             so that We can make learnable embeddings; cf below 

# careful, this assumes that transform is not trainable 
# This Lux wrapper should go into ACEbase or Polynomials4ML 
# but for now this is a convenient prototype implementation
struct GenericEmbedding{TIN, TOUT, TT, TB} <: AbstractExplicitLayer 
   transform::TT
   B::TB
   maxlen::Int 
end

GenericEmbedding(TIN, TOUT, transform, B, maxlen) = 
      GenericEmbedding{TIN, TOUT, typeof(transform), typeof(B)}(
         transform, B, maxlen
      )

Base.length(l::GenericEmbedding) = length(l.B)

function (l::GenericEmbedding)(X, ps, st)
   nX = length(X) 
   x = st.x  
   P = st.P 
   @assert length(x) >= nX
   @assert size(P, 1) >= nX

   # transform input to correct format 
   @simd ivdep for i = 1:nX 
      @inbounds x[i] = l.transform(X[i])
   end
   # now evaluate the embedding 
   Polynomials4ML.evaluate!(P, l.B, (@view x[1:nX]))
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

struct SimpleRtMEmbedding{T, TTR, TR} <: AbstractExplicitLayer
   r_trans::TTR
   r_embed::TR
   maxlen::Int 
end

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
   nX = length(X) 
   r = @view st.r[1:nX]
   tM = st.tM 
   R = st.R 
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

# function learnable_simple_RtM_embedding(T, r_trans, r_embed, nmax; maxlen=200)
#    l_R0 = 

# end
