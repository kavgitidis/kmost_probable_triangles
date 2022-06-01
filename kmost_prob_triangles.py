import argparse
import decimal
import distutils.util
import pprint
import time
from itertools import islice

import pyspark

parser = argparse.ArgumentParser()
parser.add_argument("--k", default=20, help="Number of triangles to return")
parser.add_argument("--path", default="Datasets/Youtube.csv", help="Path to dataset")
parser.add_argument("--parts", default=16, help="Number of partitions to use")
parser.add_argument("--cores", default="*", help="Number of cores to use")
parser.add_argument("--sep", default=",", help="Separator for dataset, use word SPACE for ' '")
parser.add_argument("--header", default="True", help="Header inclusion")

args = parser.parse_args()

if args.k:
    k = int(args.k)
if args.path:
    path = args.path
if args.cores:
    cores = args.cores
if args.parts:
    parts = int(args.parts)
if args.sep:
    sep = args.sep
if args.sep.lower() == "space":
    sep = " "
if args.header:
    header = args.header

now = time.time()
sc = pyspark.SparkContext(f"local[{cores}]", "KMostProbTriangles")
sc.setLogLevel("ERROR")

lines = sc.textFile(path, parts)
if bool(distutils.util.strtobool(header)):
    lines = lines.mapPartitionsWithIndex(
        lambda idx, it: islice(it, 1, None) if idx == 0 else it
    )
edges_rdd = lines.map(lambda l: (tuple(l.split(f"{sep}"))))
rdd = lines.map(lambda l: (tuple(l.split(f"{sep}")[:2])))

rdd = rdd.map(lambda x: (x[0], x[1]) if x[0] < x[1] else (x[1], x[0]))
edges_rdd = edges_rdd.map(lambda x: ((x[0], x[1]), float(x[2])) if x[0] < x[1] else ((x[1], x[0]), float(x[2])))
bc_edges = sc.broadcast(dict(edges_rdd.collect()))

neighbours = rdd.groupByKey().map(lambda x: (x[0], list(set(list(x[1])))))
bc_nb = sc.broadcast(dict(neighbours.collect()))

vertex_nb = rdd.map(lambda x: (x[0], (x[1], bc_nb.value.get((x[0])))))
vertex_nb = vertex_nb.map(lambda x: (x[1][0], (x[0], x[1][1])))
vertex_nb = vertex_nb.map(lambda x: (x[0], (x[1], bc_nb.value.get((x[0]))))).filter(lambda x: x[1][1] is not None)

triangle_combs = vertex_nb.map(lambda x: ((x[0], x[1][0][0]), list(set(x[1][0][1]) & set(x[1][1]))))
triangles = triangle_combs.flatMap(
    lambda x: (sorted(list(x[0]) + [w]) for w in x[1]))

triangles_probs = triangles.map(
    lambda x: (tuple(x),
               bc_edges.value.get((x[0], x[1])) * bc_edges.value.get((x[0], x[2])) * bc_edges.value.get(
                   (x[1], x[2]))))

kmost_prob_triangles = triangles_probs.takeOrdered(k, key=lambda x: -x[1])
pp = pprint.PrettyPrinter()
pp.pprint(kmost_prob_triangles)

later = time.time()
diff = format(decimal.Decimal(later - now), "7.2f")
print(f"Total script time: {diff} seconds")
sc.stop()
