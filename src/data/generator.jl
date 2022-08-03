
module DatasetGenerator
using Pkg
include("./reader.jl")
include("./transforms.jl")
using Statistics, LinearAlgebra, StaticArrays

using .DataReader, .DataTransformer

function parallel_transform_data(four_momentum::Vector)
    transformed_vars = Vector{Vector}[]
    Threads.@threads for momentum in four_momentum
        push!(transformed_vars, transform2hyp(momentum))
    end
    return transformed_vars
end

secuential_transform_data(four_momentum::Vector) = transform2hyp(four_momentum)

function transform_data(four_momentum::Vector, labels::Vector)
    transformed_vars = Vector{SVector{5,Float64}}[]
    ordered_labels = []
    n_limit = length(four_momentum)
    Threads.@threads for i = 1:n_limit
        push!(transformed_vars, transform2hyp(four_momentum[i]))
        push!(ordered_labels, transform2hyp(labels[i]))
    end
    return transformed_vars, ordered_labels

end

function read_data(dataset::String, filename::String; n_limit::Int=-1)::Tuple
    if dataset == "TQ"
        four_momentum, labels = read_TQ(filename; n_limit=n_limit)
    elseif dataset == "QG"
        four_momentum, labels = read_QG(filename; n_limit=n_limit)
    end
    four_momentum, labels
end

function read_transformed(dataset::String, filename::String; n_limit::Int=-1)::Tuple
    if dataset == "TQ"
        four_momentum, labels = read_TQ(filename; n_limit=n_limit)
    elseif dataset == "QG"
        four_momentum, labels = read_QG(filename; n_limit=n_limit)
    end
    parallel_transform_data(four_momentum), labels
end

export read_data, parallel_transform_data, secuential_transform_data, read_transformed

end