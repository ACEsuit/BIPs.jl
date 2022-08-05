using Test, BIPs

include("inv_test.jl")
include("safety_test.jl")

using .InvarianceTester, .PhysicsSafetyTester

sample_data_path = joinpath(BIPs.artifact("bips_test"), "toptagging_sample.h5")
sample_jets, sample_labels = BIPs.read_data("TQ", sample_data_path)
sample_hyp_jets = BIPs.data2hyp(sample_jets)

@testset "BIPs.jl" begin
    @test permutation_invariance_test(sample_hyp_jets)
    @test boost_invariance_test(sample_hyp_jets)

    @test ir_safety_test(sample_hyp_jets)
    @test collinear_safety_test(sample_hyp_jets)
end
