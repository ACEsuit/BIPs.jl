module BIPs
using Reexport
using LazyArtifacts

# Buider for the basis
include("modules/polynomials.jl")
@reexport using .BiPolynomials

# Reader for data
include("data/generator.jl")
@reexport using .DatasetGenerator

# the new lux and ACEcore based implementation 

include("lux.jl")

# a little hack to load BIPs artifacts from anywhere? 
using LazyArtifacts
artifact(str) = (@artifact_str str)

end
