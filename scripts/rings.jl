"""
27.06.23
@tcnicholas
Compute ring statistics for a given graph.
    
This script uses the input files generated with the Python script 
(scripts/rings.py) to compute ring statistics. For usage, see the tasks_si.py 
script in the main directory.
"""


using PeriodicGraphs
using DelimitedFiles
using Graphs


"""
    compute_rings(path2graph::String, cutoff::String="5.20", depth::Int64=10)

Compute ring statistics for a given graph and optionally save the ring 
structures as XYZ files.

# Arguments
- `path2graph::String`: The path to the graph data files (frac.txt, 
    edges_rcut.txt, and images_rcut.txt) and where the ouput will be placed.
- `cutoff::String="5.20"`: The cutoff distance to be used. Default is "5.20".
- `depth::Int64=10`: The maximum depth to be used in ring search. Default is 10.

# Outputs
This function saves ring statistics in `path2graph/rings_statistics_rcut<cutoff>.txt` 
and `path2graph/strongrings_statistics_rcut<cutoff>.txt`.

# Example
```julia
compute_rings("/path/to/graph/data")
"""
function compute_rings(path2graph::String,cutoff::String="5.20",depth::Int64=10)

    # read in the raw data to construct the graph.
    pos = readdlm(joinpath(path2graph,"frac.txt"), Float64)
    edges = readdlm(joinpath(path2graph, "edges_rcut$cutoff.txt"), Int)
    images = readdlm(joinpath(path2graph, "images_rcut$cutoff.txt"), Int)

    # reformat for PeriodicGraphs package.
    p = eachrow(pos) |> collect
    A = eachrow(edges) |> collect
    B = eachrow(images) |> collect

    # redefine connectivity using PeriodicEdges.
    pedges = [PeriodicEdge3D(A[i][1], A[i][2], B[i]) for i in eachindex(A)]
    g = PeriodicGraph3D(pedges)

    # iterate through all rings and statistics.
    rcount = count_rings(rings(g, depth), depth)
    write_results(path2graph, "rings_statistics_rcut$cutoff.txt", rcount, 
        "Max ring length = ")

    # iterate through all STRONG rings and statistics.
    rcount = count_rings(strong_rings(g, depth), depth)
    write_results(path2graph, "strongrings_statistics_rcut$cutoff.txt", rcount, 
        "Max strong ring length = ")
end


"""
    count_rings(g, depth::Int64)

Count the number of rings of different sizes in a graph up to a given depth.

# Arguments
- `g`: A graph object of type PeriodicGraph3D.
- `depth::Int64`: The maximum size of rings to search for.

# Returns
- `rcount`: A vector of integers where `rcount[i]` is the number of rings of size `i`.

# Example
```julia
rcount = count_rings(g, 10)
"""
function count_rings(ring_function, depth)
    rcount = zeros(Int64, depth*2+3)
    for r in sort!(first(ring_function))
        sizeofring = length(r)
        rcount[sizeofring] += 1 
    end
    return rcount
end


"""
    write_results(outpath::String, filename::String, rcount::Vector{Int64}, 
        message::String)
    
Write ring statistics to a text file.

# Arguments
- `outpath::String`: The directory where the file will be saved.
- `filename::String`: The name of the file.
- `rcount::Vector{Int64}`: A vector of integers where `rcount[i]` is the number 
    of rings of size `i`.
- `message::String`: A message to be written at the top of the file.

# Example
```julia
write_results("/path/to/output","rings_out.txt",rcount,"Max ring length = 21")
"""    
function write_results(outpath, filename, rcount, message)
    fstring = string(message, 2*length(rcount)+3, "\n")
    for (rc, c) in enumerate(rcount)
        fstring = string(fstring, rc, "\t", c, "\n")
    end
    open(joinpath(outpath, filename), "w") do io
        write(io, fstring)
    end
end



