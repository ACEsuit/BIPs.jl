using Test, BIPs


@testset "BIPs.jl" begin
    @testset "Invariance" begin include("inv_test.jl"); end
    @testset "Safety" begin include("safety_test.jl"); end
end
