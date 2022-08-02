using BIPs
using Documenter

DocMeta.setdocmeta!(BIPs, :DocTestSetup, :(using BIPs); recursive=true)

makedocs(;
    modules=[BIPs],
    authors="Christoph Ortner <c.ortner@warwick.ac.uk> and contributors",
    repo="https://github.com/cortner/BIPs.jl/blob/{commit}{path}#{line}",
    sitename="BIPs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cortner.github.io/BIPs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cortner/BIPs.jl",
    devbranch="main",
)
