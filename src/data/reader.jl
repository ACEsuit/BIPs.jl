module DataReader
using HDF5, H5Zblosc, H5Zbzip2, H5Zlz4, H5Zzstd, StaticArrays

function _get_data(hdf_table, n_limit; n_particles=200)::Tuple{Vector,Vector}
    arr = Vector{SVector{4,Float64}}[]
    labels = []
    for jet = 1:n_limit
        jet_vals = SVector{4,Float64}[]
        for (i, particle) = enumerate(1:4:4*n_particles)
            p = SVector{4}(hdf_table[jet].values_block_0[particle:particle+3])
            if p[1] > 0.0
                push!(jet_vals, p)
            end
        end
        if length(jet_vals) > 1
            push!(arr, jet_vals)
            push!(labels, hdf_table[jet].values_block_1[2])
        end
    end
    arr, labels
end


"""Read a TopQuark file in the dessired_path"""
function read_TQ(filepath::String; n_limit::Int=-1)
    fid = h5open(filepath, "r")
    table_data = read(fid, "table")
    close(fid)
    if n_limit == -1
        n_limit = length(table_data["table"])
    end
    _get_data(table_data["table"], n_limit)
end

export read_TQ
end