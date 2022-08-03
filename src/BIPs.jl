module BIPs

# Write your package code here.
# Buider for the basis
include("modules/polynomials.jl")
@reexport using bip.BiPolynomials

# Reader for data
include("data/generator.jl")
@reexport using bip.DatasetGenerator

end
