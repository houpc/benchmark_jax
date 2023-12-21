using FeynmanDiagram

order = 4
# spin = 2
# para = GV.diagPara(SigmaDiag, false, spin, order, [NoHartree])
# parquet_builder = Parquet.build(para)
# diag = parquet_builder.diagram
dict_graphs = GV.diagdict_parquet(:sigma, [(order, 0, 0)])

leafnum, leafmap = Compilers.compile_python(dict_graphs[(order, 0, 0)][1], :jax, "graphfunc$(order)00.py")

println(dict_graphs[(order, 0, 0)][2])
println(leafnum)