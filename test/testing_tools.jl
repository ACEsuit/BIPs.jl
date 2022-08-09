
using BIPs

sample_data_path = joinpath(BIPs.artifact("bips_test"), "toptagging_sample.h5")
sample_jets, sample_labels = BIPs.read_data("TQ", sample_data_path)
sample_hyp_jets = BIPs.data2hyp(sample_jets)

function bip_data(dataset_jets)
   storage = zeros(length(dataset_jets), length(specs))
   for i = 1:length(dataset_jets)
       storage[i, :] = f_bip(dataset_jets[i])
   end
   storage
end
