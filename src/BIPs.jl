module BIPs
using Reexport

# Write your package code here.
# Buider for the basis
include("modules/polynomials.jl")
@reexport using BIPs.BiPolynomials

# Reader for data
include("data/generator.jl")
@reexport using BIPs.DatasetGenerator

end
