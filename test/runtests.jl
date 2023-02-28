using Test, BIPs


@testset "BIPs.jl" begin
    @testset "Invariance" begin include("inv_test.jl"); end
    @testset "Safety" begin include("safety_test.jl"); end
    @testset "Lux-1" begin include("test_lux.jl"); end
    @testset "Lux-2" begin include("test_lux2.jl"); end
    @testset "Lux-3" begin include("test_lux_trainable.jl"); end
end
