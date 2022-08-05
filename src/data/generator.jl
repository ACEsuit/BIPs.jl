
module DatasetGenerator

include("./reader.jl")
include("./transforms.jl")
using Statistics, LinearAlgebra, StaticArrays

using .DataReader, .DataTransformer

secuential_transform_data(four_momentum::Vector) = transform2hyp(four_momentum)

function read_data(dataset::String, filename::String; n_limit::Int=-1)::Tuple
    if dataset == "TQ"
        four_momentum, labels = read_TQ(filename; n_limit=n_limit)
    elseif dataset == "QG"
        four_momentum, labels = read_QG(filename; n_limit=n_limit)
    end
    four_momentum, labels
end


export read_data, transform2hyp, data2hyp

end