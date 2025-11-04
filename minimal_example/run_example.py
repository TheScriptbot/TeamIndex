from TeamIndex import evaluation as eva
from TeamIndex import creation as crt


import numpy as np
import pandas as pd

from pathlib import Path

## create a simple uniform dummy data set
n = 400_000
columns = ["A","B","C","D","E","F","G","H","I","J"]


# note: make sure the data generated here is also used for the index created later, otherwise there will be a missmatch when checking results.
target_path = Path("./uniform_toy_data.parquet")  # will be roughly ~33MB in size


if not target_path.exists():
    table = pd.DataFrame(np.random.uniform(0, 100, n * len(columns)).reshape(n,len(columns)), columns=columns)
    table.to_parquet(target_path)
else:
    table = pd.read_parquet(target_path)

## create the index from the dummy data using the metadata already stored in the .json file.
## In the example, we group attributes A-J into the Teams [['A', 'E', 'C'], ['J', 'D', 'G'], ['B', 'I'], ['F', 'H']] with 5 bins per dimension.
## Note:    The toy_index.json file already contains quantiles for the dataset, which would ideally be recomputed before creating the index.
##          You can use "from TeamIndex.creation import determine_quantiles, create_configs"
crt.index_table("./toy_index.json", table=table)  # will not overwrite existing index data by default

## Open the index, it can now be queried
index = eva.TeamIndex("./toy_index.json")

## Example queries
# select just the very first cell in one of the Team indices, which spans roughly the interval [[0,19],[0,19],[0,19]]
# Note: only simple conjunctive WHERE clauses are supported right now
query1 = "A < 19 and E < 19 and C < 19"
query2 = "A < 19 and E < 19 and C < 19 and B < 19" # also restricts one attribute of a second Team, leading to index intersection
query3 = "A < 38 and E < 38 and C < 38 and B < 38 and C < 19" # selects multiple bins per Team for slightly more complex intersection

## use pandas query mechanism to produce reference results
ref_result_q1 = set(table.query(query1).index)
ref_result_q2 = set(table.query(query2).index)
ref_result_q3 = set(table.query(query3).index)

## evaluate indices using index intersection (if necessary). The result is a list of qualifying tuple IDs
res1 = index.run_query(query1)
res2 = index.run_query(query2)
res3 = index.run_query(query3)

## check results. The actual result is likely to be a bit smaller, depending on query selectivity, data distribution and binning.
## Index quality in this setup generally degrades with inherent (relative) result cardinality, and highly-selective queries are more accurately resolved
print("Results are correct:")
print(ref_result_q1.issubset(res1[0]))
print(ref_result_q2.issubset(res2[0]))
print(ref_result_q3.issubset(res3[0]))

## Execution of "run_query" can be manually influenced, allowing for optimizations. The current default strategy is very inefficient, a proper optimizer is WIP
## The returned object also contains some statistics that can be used by the optimizer, such as access cardinalities.
## The object is a list of pairs [(team_name, optimization_options_dictionary)], sorted in ascending order of access volume
manual_optimizations = index.prepare_optimization(query=query3)  # TODO: default strategy is still "union first"

## The following function was used as the default optimization strategy for most experiments:

def optimize(mopts):
    assert(len(mopts) >= 1), "Empty result?"

    ## "expand" the first Team, i.e., combine each bin/list from the smallest Team's result with every other list (from other Teams)
    ## Without expansion, every Team's is unified first and intersection happens last, which can be suboptimal
    mopts[0][1]["is_expanded"] = True
    if mopts[0][1]["max_group_count"] > 128:
        mopts[0][1]["group_count"] = eva.po2_near_sqrt(mopts[0][1]["max_group_count"])
    limit = 128
    if mopts[0][1]["group_count"] < 16 and len(mopts) > 1:
        mopts[1][1]["is_expanded"] = True
        limit = 16
    for i in range(1, len(mopts)):
        if mopts[i][1]["max_group_count"] > limit:
            mopts[i][1]["group_count"] = min(eva.po2_near_sqrt(mopts[i][1]["max_group_count"]), limit)

    return mopts


res3_optmized = index.run_query(query3, manual_optimizations=optimize(manual_optimizations))
print("Checking last result, too:")
print(ref_result_q3.issubset(res3_optmized[0]))

print("You can inspect the index metadata, e.g., the first Team index's 5x5x5-many leaf cardinalities via")
print("index.cardinalities[\"A-E-C\"].shape")