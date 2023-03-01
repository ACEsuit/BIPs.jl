module BiPolynomials
include("modules.jl")
using .Modules: AABasis, ChebBasis, TrigBasis, ASpec, ABasis, BasisSelector, HermBasis,
   gensparse, isadmissible, level, BasisSelector, TrigBasisNA


function generate_spec_A(inds_r, inds_θ, inds_y, Bsel)
   spec = ASpec[]
   for k in inds_r, l in inds_θ, n in inds_y
      b = ASpec(k, l, n)
      if isadmissible(Bsel, b)
         push!(spec, b)
      end
   end

   levels = level.(Ref(Bsel), spec)
   # Sort the elements in the sparse basis
   σ = sortperm(levels)
   spec = spec[σ]
   levels = levels[σ]

   return spec, levels 
end

function generate_spec_AA(spec_A, levels, Bsel)

   function tup2b(vv)
      iz = findlast(isequal(0), vv)
      if isnothing(iz)
         return [spec_A[v] for v in vv]
      end
      if iz == length(vv)
         return ASpec[]
      end
      @inbounds b = [spec_A[vv[i]] for i = iz+1:length(vv)]
      return b
   end

   function filter_lorcyl(bb)
      if length(bb) == 0
         return true
      end
      # Filtering to get rotational invariance around jet axis 
      angle_invariance = sum(b.l for b in bb) == 0
      # Filtering index to get boosting invariance
      boost_invariance = sum(b.n for b in bb) == 0
      return angle_invariance && boost_invariance
   end

   # Generate a sparse basis pointing into the A basis 
   AAspec1 = gensparse(; NU = Bsel.order,
                         maxvv = [length(spec_A) for _ = 1:Bsel.order],
                         tup2b = tup2b,
                         filter = filter_lorcyl,
                         admissible = bb -> isadmissible(Bsel, bb),
                         ordered = true)

   function fix_vv(vv)
      ww = reverse(vv)
      iz = findfirst(isequal(0), ww)
      if isnothing(iz)
         return ww 
      else
         return ww[1:iz-1]
      end
   end 

   return map(fix_vv, AAspec1)
end


"""
`build_ip` construct the sparse basis to embed the jet_vals
All arguments are keyword arguments (with defaults):
* `order`: Maximum order of the basis. Default to ν = 3
* `levels`: Maximum level of the sparse basis. Defaul to Γ = 6
*  `n_pt`: Numenber of terms in the basis for the Rk basis in the transverse 
momentum embedding. Default to n_pt = 3
* `n_eta`: Number of terms in the basis for the Angular basis. Default to n_eta = 3
* `n_phi`: Number of terms in the basis for the Rappidity basis. Default to n_phi = 3
"""
function build_ip(; order::Integer=3, levels::Integer=6,
                    n_pt::Integer=5, n_th::Integer=3, n_y::Integer=3)
   Rk = ChebBasis{:r}(n_pt)     # Define the basis Rk(r) for the transverse momentum
   Tl = TrigBasisNA(n_th)       # Tl(θ) is the basis for the polar angle
   Vn = TrigBasis(n_y)          # Defining the basis for η (pseudorapidity)


   Bsel = BasisSelector(; order=order, levels=levels)
   spec_A, levels_A = generate_spec_A(eachindex(Rk), eachindex(Tl), eachindex(Vn), Bsel)
   # And build the basis at that specification
   Abas = ABasis(Rk, Tl, Vn, spec_A)

   # generate the AA basis as a list of vectors 
   AAspec1 = generate_spec_AA(spec_A, levels_A, Bsel)

   # now convert it into the format that AABasis wants 
   orders = zeros(Int, length(AAspec1))
   AAspec = zeros(Int, Bsel.order, length(AAspec1))
   for (n, vv) in enumerate(AAspec1)
      ord = length(vv) 
      AAspec[1:ord, n] .= vv 
      orders[n] = ord 
   end

   # we can now construct an AA basis 
   basis = AABasis(Abas, AAspec, orders)
   return basis, orders
end

export build_ip
###############################################
end
