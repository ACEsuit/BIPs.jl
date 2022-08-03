module BiPolynomials
include("modules.jl")
using .Modules: AABasis, ChebBasis, TrigBasis, ASpec, ABasis, BasisSelector, HermBasis,
   gensparse, isadmissible, level, BasisSelector, TrigBasisNA


@fastmath function build_ip(; order::Integer=3, levels::Integer=6,
   n_pt::Integer=5, n_th::Integer=3, n_y::Integer=3
)
   Rk = ChebBasis{:r}(n_pt)   # Define the basis Rk(r) for the radial coordinate (transverse momentum) 
   Tl = TrigBasisNA(n_th)       # Tl(θ) is the basis for the polar angle
   Vn = TrigBasis(n_y)       # Defining the basis for η (pseudorapidity)

   #? Should we introduce weights?
   Bsel = BasisSelector(; order=order, levels=levels)
   spec = ASpec[]
   for k in eachindex(Rk), l in eachindex(Tl), n in eachindex(Vn)
      b = ASpec(k, l, n)
      if isadmissible(Bsel, b)
         push!(spec, b)
      end
   end
   # sort the A basis spec by levels (degrees)

   # Here Fixing the spec element of the Bsel basis
   levels = level.(Ref(Bsel), spec)
   # Then we must do the sorting 
   σ = sortperm(levels)
   spec = spec[σ]
   levels = levels[σ]
   # And build the basis at that specification
   Abas = ABasis(Rk, Tl, Vn, spec)
   function tup2b(vv, spec)
      iz = findlast(isequal(0), vv)
      if isnothing(iz)
         return [spec[v] for v in vv]
      end
      if iz == length(vv)
         return ASpec[]
      end
      @inbounds b = [spec[vv[i]] for i = iz+1:length(vv)]
      b
   end


   function filter_lorcyl(bb)
      if length(bb) == 0
         return true
      end
      # Filtering to get rotational invariance around jet axis 
      angle_invariance = sum(b.l for b in bb) == 0
      # Filetring index to get hyperbolic rotational invariance
      boost_inariance = sum(b.n for b in bb) == 0
      return (angle_invariance && boost_inariance)
   end
   # We generate a sparse basis pointing into the A basis 
   AAspec1 = gensparse(; NU=Bsel.order,
      maxvv=[length(spec) for _ = 1:Bsel.order],
      tup2b=vv -> tup2b(vv, spec),
      filter=filter_lorcyl,
      admissible=bb -> isadmissible(Bsel, bb),
      ordered=true)

   orders = zeros(Int, length(AAspec1))
   AAspec = zeros(Int, Bsel.order, length(AAspec1))

   # Rewritting the characteristics of the AA 
   for (n, vv) in enumerate(AAspec1)
      iz = findlast(isequal(0), vv)
      AAspec[:, n] = reverse(vv)
      if isnothing(iz)
         orders[n] = Bsel.order
      else
         orders[n] = Bsel.order - iz
      end
   end
   # we can now construct an AA basis 
   basis = AABasis(Abas, AAspec, orders)
   return basis, orders
end

export build_ip
###############################################
end
