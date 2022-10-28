
module Modules
using StaticArrays, OffsetArrays

###################################################
# Tensor Basis
###################################################

struct ASpec
    k::Int  # r
    l::Int  # θ
    n::Int  # v
end

struct ABasis{TR,TT,TV}
    bR::TR # r basis - k
    bT::TT # θ basis - l
    bV::TV # y basis - n
    spec::Vector{ASpec}
end

struct AABasis{TAB}
    Abasis::TAB
    spec::Matrix{Int}
    ords::Vector{Int}
end

Base.length(basis::ABasis) = length(basis.spec)


###################################################
# Trigonometric Basis
###################################################
struct TrigBasis
    maxL::Int
end
Base.length(basis::TrigBasis) = 2 * basis.maxL + 1
Base.eachindex(basis::TrigBasis) = -basis.maxL:basis.maxL
# ------------- Evaluating the basis in an specific angle
function (basis::TrigBasis)(θ)
    maxL = basis.maxL
    T = ComplexF64[exp(im * l * θ) for l = -maxL:maxL]
    return OffsetArray(T, -maxL:maxL)
end


###################################################
# Trigonometric Basis: No Angles
###################################################
struct TrigBasisNA
    maxL::Int
end
Base.length(basis::TrigBasisNA) = 2 * basis.maxL + 1
Base.eachindex(basis::TrigBasisNA) = -basis.maxL:basis.maxL
# ------------- Evaluating the basis in an specific angle
function (basis::TrigBasisNA)(cosθ, sinθ)
    maxL = basis.maxL
    T = [(cosθ + im * sinθ)^l for l = -maxL:maxL]
    return OffsetArray(T, -maxL:maxL)
end


###################################################
# Radial Basis: Cheb
###################################################
struct ChebBasis{SYM}
    maxn::Int
end

Base.length(basis::ChebBasis) = basis.maxn + 1

Base.eachindex(basis::ChebBasis) = 0:basis.maxn
# ------------- Evaluating the basis in an specific SYM index
# For evaluating the cheb_basis using the η variable
(basis::ChebBasis{SYM})(particle) where {SYM} = basis(particle[SYM])


function (basis::ChebBasis)(x::T) where {T<:Real}
    maxn = basis.maxn
    B = OffsetArray(zeros(T, maxn + 1), 0:maxn)
    B[0] = 1
    if maxn == 0
        return B
    end
    B[1] = x
    if maxn == 1
        return B
    end
    for n = 2:maxn
        @inbounds B[n] = 2 * x * B[n-1] - B[n-2]
    end
    return B
end


###################################################
# Radial Basis: Hermite Basis
###################################################
struct HermBasis{SYM}
    maxn::Int
end

Base.length(basis::HermBasis) = basis.maxn + 1

Base.eachindex(basis::HermBasis) = 0:basis.maxn
# ------------- Evaluating the basis in an specific SYM index
# For evaluating the cheb_basis using the η variable
(basis::HermBasis{SYM})(particle) where {SYM} = basis(particle[SYM])

function hermite_transformer(x::T; a=5.0, b=0.01, c=4.0) where {T<:Real}
    log((x + b) / c) / a
end


function (basis::HermBasis)(x::T) where {T<:Real}
    maxn = basis.maxn
    x = hermite_transformer(x)
    x = x / 1.41
    B = OffsetArray(zeros(T, maxn + 1), 0:maxn)
    B[0] = 1
    if maxn == 0
        return B
    end
    B[1] = 2 * x
    if maxn == 1
        return B
    end
    for n = 2:maxn
        @inbounds B[n] = (2^(-n / 2)) * (2.0 * x * B[n-1] - 2.0 * (n - 1) * B[n-2])
    end
    return B
end


#####################################################################


#####################################################################
# The Invariant Polynomials
#####################################################################

function _addinto!(A::Vector, basis::ABasis, x)
    # x = SVector(r, cosθ, sinθ, y, tM)
    @inbounds Rk = basis.bR((log(x[1]) + 4.7) / 6)
    @inbounds Tl = basis.bT(x[2], x[3])
    @inbounds Vn = basis.bV(x[4])
    @inbounds Um = x[end]
    for (i, b) in enumerate(basis.spec)
        @inbounds A[i] += Rk[b.k] * Tl[b.l] * Vn[b.n] * Um
    end
    return nothing
end

function (basis::ABasis)(X)
    A = zeros(ComplexF64, length(basis))
    @inbounds for x in X
        _addinto!(A, basis, x)
    end
    return A
end

# -------------- Product Basis

@fastmath function (basis::AABasis)(X)
    A = basis.Abasis(X)
    spec = basis.spec
    ords = basis.ords
    @inbounds ret = real([prod(A[spec[t, iAA]] for t = 1:ords[iAA]; init=one(eltype(A)))
                          for iAA = 1:length(basis)])
    ret
end

Base.length(basis::AABasis) = size(basis.spec, 2)


# -------------- Basis generation
struct BasisSelector
    order::Int
    weights::Dict{Symbol,Float64}
    levels::Vector{Float64}
end
maxlevel(bsel::BasisSelector) = maximum(bsel.levels)


function BasisSelector(; order::Integer=3,
    weights=nothing, levels=nothing)
    if levels isa Number
        levels = [Float64(levels) for _ = 1:order]
    else
        levels = Float64.(levels)
        @assert levels isa Vector{Float64}
        @assert length(levels) == order
    end

    if isnothing(weights)
        weights = Dict(:k => 1.0, :l => 1.0, :n => 1.0, :m => 1.0)
    else
        @assert weigths isa Dict{Symbol,Float64}
    end

    return BasisSelector(order, weights, levels)
end

level(bsel::BasisSelector, b::ASpec) =
    sum(bsel.weights[sym] * abs(getproperty(b, sym))
        for sym in (:k, :l, :n))

level(bsel::BasisSelector, bb::AbstractVector{ASpec}) =
    sum(level.(Ref(bsel), bb))

isadmissible(bsel::BasisSelector, b::ASpec) =
    level(bsel, b) <= maxlevel(bsel)

isadmissible(bsel::BasisSelector, bb::AbstractVector{ASpec}) =
    isempty(bb) ? true : (level(bsel, bb) <= bsel.levels[length(bb)])


#######################################################################################3
# Building symmetrization of the basis



"""
`_gensparse` : function barrier for `gensparse`
"""
function _gensparse(::Val{NU}, tup2b, admissible, filter, INT, ordered,
    minvv, maxvv) where {NU}
    @assert INT <: Integer

    lastidx = 0
    vv = @MVector zeros(INT, NU)
    for i = 1:NU
        vv[i] = minvv[i]
    end

    spec = SVector{NU,INT}[]
    orig_spec = SVector{NU,INT}[]

    if NU == 0
        if all(minvv .== 0) && admissible(vv) && filter(vv)
            push!(spec, SVector(vv))
        end
        return spec
    end

    while true
        isadmissible = true
        if any(vv .> maxvv)
            isadmissible = false
        else
            bb = tup2b(vv)
            isadmissible = admissible(bb)
        end

        if isadmissible
            if filter(bb)
                push!(spec, SVector(vv))
                push!(orig_spec, copy(SVector(vv)))
            end
            lastidx = NU
            vv[lastidx] += 1
        else
            if lastidx == 0
                error("""lastidx == 0 should never occur; this means that the
                         smallest basis function is already inadmissible and therefore
                         the basis is empty.""")
            end
            if lastidx == 1
                break
            end
            vv[lastidx-1] += 1
            if ordered   # ordered tuples (permutation symmetry)
                vv[lastidx:end] .= vv[lastidx-1]
            else         # unordered tuples (no permutation symmetry)
                vv[lastidx:end] .= 0
            end
            lastidx -= 1
        end
    end

    if ordered
        @assert all(issorted, orig_spec)
        @assert length(unique(orig_spec)) == length(orig_spec)
    end

    return spec
end


"""
`gensparse(...)` : utility function to generate high-dimensional sparse grids
which are downsets.
All arguments are keyword arguments (with defaults):
* `NU` : maximum correlation order
* `minvv = 0` : `minvv[i] gives the minimum value for `vv[i]`
* `maxvv = Inf` : `maxvv[i] gives the maximum value for `vv[i]`
* `tup2b = vv -> vv` :
* `admissible = _ -> false` : determines whether a tuple belongs to the downset
* `filter = _ -> true` : a callable object that returns true of tuple is to be kept and
false otherwise (whether or not it is part of the downset!) This is used, e.g.
to enforce conditions such as ∑ lₐ = even or |∑ mₐ| ≦ M
* `INT = Int` : integer type to be used
* `ordered = false` : whether only ordered tuples are produced; ordered tuples
correspond to  permutation-invariant basis functions
"""
gensparse(; NU::Integer=nothing,
    minvv=[0 for _ = 1:NU],
    maxvv=[Inf for _ = 1:NU],
    tup2b=vv -> vv,
    admissible=_ -> false,
    filter=_ -> true,
    INT=Int,
    ordered=false) =
    _gensparse(Val(NU), tup2b, admissible, filter, INT, ordered,
        SVector(minvv...), SVector(maxvv...))


export AABasis, ChebBasis, TrigBasis, ASpec, ABasis, BasisSelector, HermBasis,
    gensparse, isadmissible, level, BasisSelector, TrigBasisNA

end


