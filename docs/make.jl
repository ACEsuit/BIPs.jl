using BIPs
using Documenter

DocMeta.setdocmeta!(BIPs, :DocTestSetup, :(using BIPs); recursive=true)

makedocs(;
    modules=[BIPs],
    authors="Jose M Munoz <munozariasjm@gmail.com>, Christoph Ortner <christophortner0@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/BIPs.jl/blob/{commit}{path}#{line}",
    sitename="BIPs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/BIPs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/BIPs.jl",
    devbranch="main",
)
